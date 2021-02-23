import torch
import torch.nn as nn
import math
import os
import json
import pandas as pd
import sys
from utils.config import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# for baseline, fix the args
args['single_mode'] = True
args['addition_name']  = ''


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
    model_name = 'SingleTextCNN'
else:
    model_name = 'CombineTextCNN'

save_prefix = 'BSZ' + str(args['batch_size']) + 'LR' + str(args['lr']) + 'KN' + str(args['kernel_num']) + 'KSZ' + str(args['kernel_sizes']).replace('[','').replace(']','').replace(',','').replace(' ','') + 'DR' + str(args['dropout']) + 'SD' + str(args['model_seed'])

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
        lengths = [len(seq) for seq in seqs]
        max_len = max(max(lengths),max(args['kernel_sizes']))
        padded_seqs = torch.zeros(len(seqs), max_len).long() # [batch_size, max_len] and PAD_token=0
        for i, seq in enumerate(seqs):         
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs
    his, nxt, res, label = zip(*data)
    his = _pad_seqs(his)
    nxt = _pad_seqs(nxt)
    res = _pad_seqs(res)
    label = torch.tensor(label)
    return his, nxt, res, label



class TextCNN(nn.Module):
    # TextCNN: CNN-rand, CNN-static, CNN-non-static, CNN-multichannel
    def __init__(self, vocab_size, embed_dim, kernel_num, kernel_sizes, class_num, pretrain_embed, static, non_static, multichannel):
        '''
        :param vocab_size: 词表大小
        :param embed_dim: 词向量维度
        :param kernel_num: kernel数目
        :param kernel_sizes: 不同kernel size
        :param class_num: 类别数
        :param pretrain_embed: 预训练词向量
        :param static: 是否使用预训练词向量, static=True, 表示使用预训练词向量
        :param non_static: 是否微调，non_static=True,表示不微调
        :param multichannel: 是否多通道
        '''
        super(TextCNN, self).__init__()
        
        # 初始化为单通道
        channel_num = 1
        # 随机初始化词向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)      
        # 使用预训练词向量
        if static:
            self.embedding = self.embedding.from_pretrained(pretrain_embed, freeze=not non_static)    
        # 微调+固定预训练词向量
        if multichannel:
            # defalut: freeze=True, 即默认embedding2是固定的
            self.embedding2 = nn.Embedding(vocab_size, embed_dim).from_pretrained(pretrain_embed)
            channel_num = 2
        else:
            self.embedding2 = None
        # 卷积层, kernel size: (size, embed_dim), output: [(batch_size, kernel_num, h,1)] 
        self.convs = nn.ModuleList([
            nn.Conv2d(channel_num, kernel_num, (size, embed_dim)) 
            for size in kernel_sizes
        ])

    def forward(self, x):
        '''
        :params x: (batch_size, max_len)
        :return x: logits
        '''
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1) # (batch_size, 2, max_len, word_vec)
        else:
            x = self.embedding(x).unsqueeze(1) # (batch_size, 1, max_len, word_vec) 
#        try: 
        # 卷积    
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [(batch_size, kernel_num, h)]
        # 池化
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x] # [(batch_size, kernel_num)]
        # flatten
        x = torch.cat(x, 1) # (batch_size, kernel_num * len(kernel_sizes)) 
#        except:
#            logging.info('Error at {} \n\n'.format(file_name))
#            sys.exit(0)

        return x


class Model(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, kernel_num, kernel_sizes, class_num, pretrain_embed, dropout, static, non_static, multichannel, single_mode):
        super(Model, self).__init__()
        self.textcnn = TextCNN(vocab_size, 
                    embed_dim, 
                    kernel_num, 
                    kernel_sizes, 
                    class_num,
                    pretrain_embed,
                    static, 
                    non_static, 
                    multichannel)


        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(kernel_num * len(kernel_sizes) , class_num)
        self.dense = nn.Linear(kernel_num * len(kernel_sizes) * 4, kernel_num * len(kernel_sizes))
        self.dense2 = nn.Linear(kernel_num * len(kernel_sizes) * 3, kernel_num * len(kernel_sizes))
        self.single_mode = single_mode

    def forward(self, his, nxt=None, res=None):
        if self.single_mode:
            out = self.textcnn(his)
            out = self.dropout(out)
            out = self.fc(out)
            return out
        else:
            if args['addition_name'] == 'ModCNN':
                # ModCNN
                out_his = self.textcnn(his)
                out_nxt = self.textcnn(nxt)
                out_res = self.textcnn(res)
                out_HisNxt = torch.cat([out_his, out_nxt], 1)
                out_HisRes = torch.cat([out_his, out_res], 1)
                out = torch.cat([out_HisNxt, out_HisRes], 1)
                out = self.dense(out)
                out = self.dropout(out)
                out = self.fc(out)
            elif args['addition_name'] == 'ThreeDRCNN':
                # ThreeDRCNN
                out_his = self.dropout(self.textcnn(his))
                out_nxt = self.dropout(self.textcnn(nxt))
                out_res = self.dropout(self.textcnn(res))
                out = torch.cat([out_his, out_nxt, out_res], 1)
                out = self.dense2(out)
                out=  self.dropout(out)
                out = self.fc(out)
            elif args['addition_name'] == 'ThreeCNN':
                # ThreeCNN
                out_his = self.textcnn(his)
                out_nxt = self.textcnn(nxt)
                out_res = self.textcnn(res)
                out = torch.cat([out_his, out_nxt, out_res], 1)
                out = self.dense2(out)
                out=  self.dropout(out)
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

    for i, (his, nxt, res, label) in enumerate(train_loader): 
        if args['single_mode']:
            # 移动到GPU
            his = his.to(device)
            label = label.to(device)       
            # 前向计算
            logit = model(his) 
        else:
            # 移动到GPU
            his = his.to(device)
            nxt = nxt.to(device)
            res = res.to(device)
            label = label.to(device)    
            # 前向计算
            logit = model(his, nxt, res)
            
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
        for i, (his, nxt, res, label) in enumerate(val_loader):  
            if args['single_mode']:
                # 移动到GPU
                his = his.to(device)
                label = label.to(device)       
                # 前向计算
                logit = model(his)    
            else:
                # 移动到GPU
                his = his.to(device)
                nxt = nxt.to(device)
                res = res.to(device)
                label = label.to(device)             
                # 前向计算
                logit = model(his, nxt, res)
            
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
        for i, (his, nxt, res, label) in enumerate(test_loader):  
            if args['single_mode']:
                # 移动到GPU
                his = his.to(device)
                label = label.to(device)       
                # 前向计算
                logit = model(his)    
            else:
                # 移动到GPU
                his = his.to(device)
                nxt = nxt.to(device)
                res = res.to(device)
                label = label.to(device)             
                # 前向计算
                logit = model(his, nxt, res)
            
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
                                              num_workers=4)


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
                                              num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                              batch_size=args['batch_size'],
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              num_workers=4)

    # 初始化/加载模型
    if checkpoint is None:
        if args['static'] == False: embed_dim = args['embed_dim_default']
        model = Model(vocab_size=len(word_map), 
                      embed_dim=embed_dim, 
                      kernel_num=args['kernel_num'], 
                      kernel_sizes=args['kernel_sizes'], 
                      class_num=class_num,
                      pretrain_embed=pretrain_embed,
                      dropout=args['dropout'], 
                      static=args['static'], 
                      non_static=args['non_static'], 
                      multichannel=args['multichannel'],
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






def token2str(lang, data):
    # data: (batch_size, max_len)
    arr = []
    for i in range(len(data)):
        s = " ".join([lang[int(idx)] for idx in data[i] if int(idx) not in [PAD_token, SOS_token, EOS_token]])
        assert s != " "
        arr.append(s)
    assert arr != []
    return arr

def stat_acc(logits, targets):
    corrects = (torch.max(logits, 1)[1].view(targets.size()).data == targets.data).cpu()
    return corrects

def sample_test(test_data, checkpoint):
    global args
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = checkpoint['model']
    args = checkpoint['args']
    lang = Lang(os.path.join(data_path, args['data_name'], 'class_wordmap.json'))
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=args['batch_size'],
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              num_workers=4)
    # 移动到GPU
    model = model.to(device)
    # loss function
    criterion = nn.CrossEntropyLoss().to(device)
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
        all_data = pd.DataFrame(columns=['his','agent_gen','user_gen','label','correct_flag'])
        for i, (his, nxt, res, label) in enumerate(test_loader): 
            his_str = token2str(lang, his)
            nxt_str = token2str(lang, nxt)
            res_str = token2str(lang, res)
            df = pd.DataFrame(columns=['his','agent_gen','user_gen','label','correct_flag'])
            df['his'] = his_str
            df['agent_gen'] = nxt_str
            df['user_gen'] = res_str
            df['label'] = label
            if args['single_mode']:
                # 移动到GPU
                his = his.to(device)
                label = label.to(device)       
                # 前向计算
                logit = model(his)    
            else:
                # 移动到GPU
                his = his.to(device)
                nxt = nxt.to(device)
                res = res.to(device)
                label = label.to(device)             
                # 前向计算
                logit = model(his, nxt, res)
            
            # 计算整个batch上的平均loss
            loss = criterion(logit, label)
            
            # 计算准确率
            accs.update(accuracy(logit, label))
            precision, recall, f1 = f1_compute(logit, label)
            precs.update(precision)
            recs.update(recall)
            f1s.update(f1)
            losses.update(loss.item())
            corrects = stat_acc(logit, label)
            df['correct_flag'] = corrects
            all_data = all_data.append(df)

        # 计算整个测试集上的正确率
        logging.info(
            'LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f},'
            'Precision - {prec.avg:.3f}, Recall - {rec.avg:.3f}, F1 - {f1.avg:.3f}'.format(loss=losses, acc=accs, prec=precs, rec=recs, f1=f1s))
        if args['single_mode']:
            all_data.to_csv(os.path.join('output', 'Baseline_' + args['data_name'] + '_textcnn_sample.csv'))
        else:
            all_data.to_csv(os.path.join('output',  args['data_name'] + '_textcnn_sample.csv'))
    return accs.avg



if __name__ == "__main__":

    train_data, val_data, test_data  = prepare_combine_data(args['data_name'], args['nmt_name'], args['punc'])
    if args['do_train']:
        train_eval(train_data, val_data)
    if args['do_eval']:
        if args['ckpt'] != '':
            test(test_data, checkpoint=args['ckpt'], acc_flag=False)
        else:
            test(test_data, checkpoint='checkpoints/' + 'BEST_checkpoint_' + file_name + '.pt')
    if 'do_sample' in args.keys() and args['do_sample']:
        train_data, val_data, test_data  = prepare_combine_data(args['data_name'], args['nmt_name'], args['punc'])
        sample_test(test_data, args['ckpt'])


    

