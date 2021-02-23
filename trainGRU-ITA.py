import torch
import torch.nn as nn
import math
import os
import json
import sys
from utils.config import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# for GRU-ITA, fix the args
args['single_mode'] = False
args['addition_name'] = 'ModRNN'

if args['data_name'] == 'multiwoz':
    pos_label = 1
    data_path = 'data/newAgentUser/'
    args['data_name'] = 'newSpUser' # we use the name denoting the modified dataset
    from utils.utils_MultiWoz import prepare_combine_data, Lang
elif args['data_name'] == 'dailydialog':
    pos_label = 1
    args['data_name'] = 'DailyUser'
    data_path = 'data/newDailyDialog/'
    from utils.utils_DailyDialog import prepare_combine_data, Lang
elif args['data_name'] == 'ccpe':
    # since the actual meaning of the "agent" symbol used in the code is the "user" described in the paper, we need to flip the positive class label, opennmt results, dataset statistics results
    pos_label = 0
    args['data_name'] = 'CCPEAgent'
    data_path = 'data/newCCPE/'
    from utils.utils_CCPE import prepare_combine_data, Lang
else:
    logging.info('wrong input...')
    sys.exit(1)




# 其他参数
torch.cuda.manual_seed_all(args['model_seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_num = 2
print_freq = 100  # 每隔print_freq个iteration打印状态
checkpoint = None  # 模型断点所在位置, 无则None


# 数据集 + 去标点 + 模型 + 参数
if args['punc']:
    punc_name = '_punc'
else:
    punc_name = ''
data_name = args['data_name'] + punc_name

if args['single_mode']:
    model_name = 'SingleRNN'
else:
    model_name = 'CombineRNN'

save_prefix = 'BSZ' + str(args['batch_size']) + 'LR' + str(args['lr']) + 'HD' + str(args['hidden_size']) + 'DR' + str(args['dropout']) + 'BI' + str(args['birnn']) + 'SD' + str(args['model_seed'])

if args['nmt_name'] != 'origin':
    save_prefix += 'NMT' + args['nmt_name']

file_name = data_name + '_' + model_name + '_' + save_prefix + args['addition_name']
logging.info(args)
logging.info(file_name)


# 评测指标
from sklearn.metrics import precision_recall_fscore_support
def f1_compute(logits, targets):
    actuals = torch.max(logits, 1)[1].cpu().numpy()
    targets = targets.cpu().numpy()
    p,r,f,_ = precision_recall_fscore_support(targets, actuals, average='binary', pos_label=pos_label)
    return p * 100.0, r * 100.0, f * 100.0



def collate_fn(data):
    def _pad_seqs(seqs):
        lengths = torch.tensor([len(seq) for seq in seqs])
        sort_idx = torch.argsort(-lengths)
        padded_seqs = torch.zeros(len(seqs), max(lengths)).long() # [batch_size, max_len] and PAD_token=0
        for i, seq in enumerate(seqs):         
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs[sort_idx] # sorted
        lengths = lengths[sort_idx] #sorted
        return padded_seqs, lengths, sort_idx
    his, nxt, res, label = zip(*data)

    his, his_lengths, his_sort_idx = _pad_seqs(his)
    nxt, nxt_lengths, nxt_sort_idx = _pad_seqs(nxt)
    res, res_lengths, res_sort_idx = _pad_seqs(res)
    label = torch.tensor(label)
    return his, nxt, res, label, his_sort_idx, nxt_sort_idx, res_sort_idx, his_lengths, nxt_lengths, res_lengths

class EncoderRNN(nn.Module):
    def __init__(self, num_layers, hidden_size, vocab_size, pretrain_embed, embed_dim, dropout, birnn):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.birnn = birnn
        #nn.Embedding输入向量维度是字典长度,输出向量维度是词向量维度
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding = self.embedding.from_pretrained(pretrain_embed, freeze=False)
        #双向GRU作为Encoder
        self.gru = nn.GRU(self.embed_dim, self.hidden_size, self.num_layers,
                          dropout=(0 if self.num_layers == 1 else self.dropout), bidirectional=self.birnn, batch_first=True)
    
    def forward(self, seq, sort_idx, seq_len):
        unsort_idx = torch.argsort(sort_idx)
        embedded = self.embedding(seq) # batch_size, max_len, word_vec
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_len, batch_first=True)
        _, hidden = self.gru(packed)
        hidden = hidden.transpose(0,1)
        # hidden: [batch_size, num_layers*num_directions, hidden_size]
        # outputs: [batch_size, max_seq_len, hidden_size*num_directions]
        #outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        hidden = hidden[:, :self.num_layers, :][unsort_idx].squeeze(1)
        # hidden: [batch_size, num_layers, hidden_size]
        # outputs: [batch_size, max_seq_len, hidden_size]

        return hidden

class Model(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_embed, embed_dim, rnn_dr, birnn, dropout, single_mode, class_num):
        super(Model, self).__init__()
        self.encoder = EncoderRNN(
            num_layers=1,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            pretrain_embed=pretrain_embed,
            embed_dim=embed_dim,
            dropout=rnn_dr,
            birnn=birnn)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, class_num)
        self.dense = nn.Linear(hidden_size * 4, hidden_size)
        self.single_mode = single_mode
        
    
    def forward(self, his, his_sort_idx, his_lengths, nxt=None, nxt_sort_idx=None, nxt_lengths=None, res=None, res_sort_idx=None, res_lengths=None):
        if self.single_mode:
            out = self.encoder(his, his_sort_idx, his_lengths)
            out = self.dropout(out)
            out = self.fc(out)
            return out
        else:
            if args['addition_name'] == 'ModRNN':
                his = self.encoder(his, his_sort_idx, his_lengths) #[batch_size, hidden_size]
                nxt = self.encoder(nxt, nxt_sort_idx, nxt_lengths)
                res = self.encoder(res, res_sort_idx, res_lengths)
                out_HisNxt = torch.cat([his, nxt], 1)
                out_HisRes = torch.cat([his, res], 1)
                out = torch.cat([out_HisNxt, out_HisRes], 1)
                out = self.dense(out)
                out = self.dropout(out)
                out = self.fc(out)
                return out


class AverageMeter(object):
    '''
    跟踪指标的最新值,平均值,和,count
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0. #value
        self.avg = 0. #average
        self.sum = 0. #sum
        self.count = 0 #count

    def update(self, val, n=1):
        self.val = val # 当前batch的val
        self.sum += val * n # 从第一个batch到现在的累加值
        self.count += n # 累加数目加1
        self.avg = self.sum / self.count # 从第一个batch到现在的平均值



def accuracy(logits, targets):
    '''
    计算单个batch的正确率
    :param logits: (batch_size, class_num)
    :param targets: (batch_size)
    :return: 
    '''
    corrects = (torch.max(logits, 1)[1].view(targets.size()).data == targets.data).sum()
    return corrects.item() * (100.0 / targets.size(0))

def adjust_learning_rate(optimizer, current_epoch):
    '''
    学习率衰减
    '''
    frac = float(current_epoch - args['decay_epoch']) / 50
    shrink_factor = math.pow(0.5, frac)
    
    logging.info("DECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor

    logging.info("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))


def save_checkpoint(file_name, epoch, epochs_since_improvement, model, optimizer, acc, is_best):
    '''
    保存模型

    :param epoch: epoch number
    :param epochs_since_improvement: 自上次提升正确率后经过的epoch数
    :param model:  model
    :param optimizer: optimizer
    :param acc: 每个epoch的验证集上的acc
    :param is_best: 该模型参数是否是目前最优的
    '''
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc,
             'model': model,
             'optimizer': optimizer,
             'args': args}
    savename = 'checkpoint_' + file_name + '.pt'
    torch.save(state, 'checkpoints/' + savename)
    # 如果目前的checkpoint是最优的，添加备份以防被重写
    if is_best:
        torch.save(state, 'checkpoints/' + 'BEST_' + savename)


def train(train_loader, model, criterion, optimizer, epoch, vocab_size, print_freq, device):
    '''
    执行一个epoch的训练
    
    :param train_loader: DataLoader
    :param model: model
    :param criterion: 交叉熵loss
    :param optimizer:  optimizer
    :param epoch: 执行到第几个epoch
    :param vocab_size: 词表大小
    :param print_freq: 打印频率
    :param device: device
    '''
    # 切换模式(使用dropout)
    model.train()
    
    losses = AverageMeter()  # 一个batch的平均loss
    accs = AverageMeter()  # 一个batch的平均正确率
    precs = AverageMeter()
    recs = AverageMeter()
    f1s = AverageMeter()      

    for i, (his, nxt, res, label, his_sort_idx, nxt_sort_idx, res_sort_idx, his_lengths, nxt_lengths, res_lengths) in enumerate(train_loader): 
        if args['single_mode']:
            # 移动到GPU
            his = his.to(device)
            his_sort_idx = his_sort_idx.to(device)
            his_lengths = his_lengths.to(device)
            label = label.to(device)       
            # 前向计算
            logit = model(his, his_sort_idx, his_lengths) 
        else:
            # 移动到GPU
            his = his.to(device)
            his_sort_idx = his_sort_idx.to(device)
            his_lengths = his_lengths.to(device)

            nxt = nxt.to(device)
            nxt_sort_idx = nxt_sort_idx.to(device)
            nxt_lengths = nxt_lengths.to(device)

            res = res.to(device)
            res_sort_idx = res_sort_idx.to(device)
            res_lengths = res_lengths.to(device)

            label = label.to(device)    
            # 前向计算
            logit = model(his, his_sort_idx, his_lengths, nxt, nxt_sort_idx, nxt_lengths, res, res_sort_idx, res_lengths)
            
        # 计算整个batch上的平均loss
        loss = criterion(logit, label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 计算准确率
        accs.update(accuracy(logit, label))
        precision, recall, f1 = f1_compute(logit, label)
        precs.update(precision)
        recs.update(recall)
        f1s.update(f1)
        losses.update(loss.item())
        
        # 打印状态
        if i % print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {rec.val:.3f} ({rec.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                                          loss=losses,
                                                                          acc=accs,
                                                                          prec=precs,
                                                                          rec=recs,
                                                                          f1=f1s))



def validate(val_loader, model, criterion, print_freq, device):
    '''
    执行一个epoch的验证(跑完整个验证集)
    :param val_loader: 验证集的DataLoader
    :param model: model
    :param criterion: 交叉熵loss
    :param print_freq: 打印频率
    :param device: device
    :return: accuracy
    '''
    
    #切换模式
    model = model.eval()

    losses = AverageMeter()  # 一个batch的平均loss
    accs = AverageMeter()  # 一个batch的平均正确率
    precs = AverageMeter()
    recs = AverageMeter()
    f1s = AverageMeter()  
    # 设置不计算梯度
    with torch.no_grad():
        # 迭代每个batch
        for i, (his, nxt, res, label, his_sort_idx, nxt_sort_idx, res_sort_idx, his_lengths, nxt_lengths, res_lengths) in enumerate(val_loader): 
            if args['single_mode']:
                # 移动到GPU
                his = his.to(device)
                his_sort_idx = his_sort_idx.to(device)
                his_lengths = his_lengths.to(device)
                label = label.to(device)       
                # 前向计算
                logit = model(his, his_sort_idx, his_lengths) 
            else:
                # 移动到GPU
                his = his.to(device)
                his_sort_idx = his_sort_idx.to(device)
                his_lengths = his_lengths.to(device)

                nxt = nxt.to(device)
                nxt_sort_idx = nxt_sort_idx.to(device)
                nxt_lengths = nxt_lengths.to(device)

                res = res.to(device)
                res_sort_idx = res_sort_idx.to(device)
                res_lengths = res_lengths.to(device)

                label = label.to(device)    
                # 前向计算
                logit = model(his, his_sort_idx, his_lengths, nxt, nxt_sort_idx, nxt_lengths, res, res_sort_idx, res_lengths)
            
            # 计算整个batch上的平均loss
            loss = criterion(logit, label)      
            # 计算准确率
            accs.update(accuracy(logit, label))
            precision, recall, f1 = f1_compute(logit, label)
            precs.update(precision)
            recs.update(recall)
            f1s.update(f1)
            losses.update(loss.item())

            if i % print_freq  == 0:
                logging.info('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {rec.val:.3f} ({rec.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'.format(i, len(val_loader),
                                                                          loss=losses,
                                                                          acc=accs,
                                                                          prec=precs,
                                                                          rec=recs,
                                                                          f1=f1s))
        # 计算整个验证集上的正确率
        logging.info(
            'LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f},'
            'Precision - {prec.avg:.3f}, Recall - {rec.avg:.3f}, F1 - {f1.avg:.3f}'.format(loss=losses, acc=accs, prec=precs, rec=recs, f1=f1s))

    return accs.avg


def testing(test_loader, model, criterion, print_freq, device):
    '''
    执行测试
    :param test_loader: 测试集的DataLoader
    :param model: model
    :param criterion: 交叉熵loss
    :param print_freq: 打印频率
    :param device: device
    :return: accuracy
    '''
    
    #切换模式
    model = model.eval()

    losses = AverageMeter()  # 一个batch的平均loss
    accs = AverageMeter()  # 一个batch的平均正确率
    precs = AverageMeter()
    recs = AverageMeter()
    f1s = AverageMeter()
    # 设置不计算梯度
    with torch.no_grad():
        # 迭代每个batch
        for i, (his, nxt, res, label, his_sort_idx, nxt_sort_idx, res_sort_idx, his_lengths, nxt_lengths, res_lengths) in enumerate(test_loader): 
            if args['single_mode']:
                # 移动到GPU
                his = his.to(device)
                his_sort_idx = his_sort_idx.to(device)
                his_lengths = his_lengths.to(device)
                label = label.to(device)       
                # 前向计算
                logit = model(his, his_sort_idx, his_lengths) 
            else:
                # 移动到GPU
                his = his.to(device)
                his_sort_idx = his_sort_idx.to(device)
                his_lengths = his_lengths.to(device)

                nxt = nxt.to(device)
                nxt_sort_idx = nxt_sort_idx.to(device)
                nxt_lengths = nxt_lengths.to(device)

                res = res.to(device)
                res_sort_idx = res_sort_idx.to(device)
                res_lengths = res_lengths.to(device)

                label = label.to(device)    
                # 前向计算
                logit = model(his, his_sort_idx, his_lengths, nxt, nxt_sort_idx, nxt_lengths, res, res_sort_idx, res_lengths)
            
            # 计算整个batch上的平均loss
            loss = criterion(logit, label)
            
            # 计算准确率
            accs.update(accuracy(logit, label))
            precision, recall, f1 = f1_compute(logit, label)
            precs.update(precision)
            recs.update(recall)
            f1s.update(f1)  
            losses.update(loss.item())
            
            if i % print_freq  == 0:
                logging.info('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {rec.val:.3f} ({rec.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'.format(i, len(test_loader),
                                                                          loss=losses,
                                                                          acc=accs,
                                                                          prec=precs,
                                                                          rec=recs,
                                                                          f1=f1s))

        # 计算整个测试集上的正确率
        logging.info(
            'LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f},'
            'Precision - {prec.avg:.3f}, Recall - {rec.avg:.3f}, F1 - {f1.avg:.3f}'.format(loss=losses, acc=accs, prec=precs, rec=recs, f1=f1s))

    return accs.avg


def test(test_data, checkpoint, acc_flag=True):

    global args
    filename = checkpoint

    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = checkpoint['model']
    args = checkpoint['args']

    logging.info(model)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=args['batch_size'],
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              num_workers=0)


    # 移动到GPU
    model = model.to(device)
    
    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    acc = testing(test_loader=test_loader,
                              model=model,
                              criterion=criterion,
                              print_freq=print_freq,
                              device=device)

    if 'ACC' not in filename and acc_flag:
        os.rename(filename, filename[:-3] + '_ACC' + str(round(acc, 3)) + '.pt')
        os.remove(filename[:12] + filename[17:])
    return acc




def train_eval(train_data, val_data):
    '''
    训练和验证
    '''
    # 初始化best accuracy

    global checkpoint
    global args

    best_acc = 0.

    # epoch
    start_epoch = 0
    epochs_since_improvement = 0  # 跟踪训练时的验证集上的BLEU变化，每过一个epoch没提升则加1

    # 读入词表

    word_map_file = data_path + args['data_name'] +  '/class_wordmap.json'
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # 加载预训练词向量
    embed_file = data_path + args['data_name'] +  '/pretrain_embed.pt'
    embed_file = torch.load(embed_file)
    pretrain_embed, embed_dim = embed_file['pretrain'], embed_file['dim']

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=args['batch_size'],
                                              shuffle=True,
                                              collate_fn=collate_fn,
                                              num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                              batch_size=args['batch_size'],
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              num_workers=0)

    # 初始化/加载模型
    if checkpoint is None:
        model = Model(vocab_size=len(word_map), 
                      embed_dim=embed_dim, 
                      class_num=class_num,
                      pretrain_embed=pretrain_embed,
                      dropout=args['dropout'], 
                      rnn_dr=0,
                      birnn=args['birnn'],
                      hidden_size=args['hidden_size'],
                      single_mode=args['single_mode'])
    
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=args['lr'],
                                     weight_decay=args['weight_decay'])

        
    else:
        # 载入checkpoint
        checkpoint = torch.load(checkpoint, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_acc = checkpoint['acc']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        args = checkpoint['args']
    
    # 移动到GPU
    model = model.to(device)
    
    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Epochs
    for epoch in range(start_epoch, args['epochs']):
        
        # 学习率衰减
        if epoch > args['decay_epoch']:
            adjust_learning_rate(optimizer, epoch)
        
        # early stopping 如果dev上的acc在6个连续epoch上没有提升
        if epochs_since_improvement == args['improvement_epoch']:
            break
        
        # 一个epoch的训练
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              vocab_size=len(word_map),
              print_freq=print_freq,
              device=device)
        
        # 一个epoch的验证
        recent_acc = validate(val_loader=val_loader,
                              model=model,
                              criterion=criterion,
                              print_freq=print_freq,
                              device=device)
        
        # 检查是否有提升
        is_best = recent_acc > best_acc
        best_acc = max(recent_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            logging.info("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        
        # 保存模型
        save_checkpoint(file_name, epoch, epochs_since_improvement, model, optimizer, recent_acc, is_best)


if __name__ == "__main__":

    train_data, val_data, test_data  = prepare_combine_data(args['data_name'], args['nmt_name'], args['punc'])
    if args['do_train']:
        train_eval(train_data, val_data)
    if args['do_eval']:
        if args['ckpt'] != '':
            test(test_data, checkpoint=args['ckpt'], acc_flag=False)
        else:
            test(test_data, checkpoint='checkpoints/' + 'BEST_checkpoint_' + file_name + '.pt')


    

