import torch
import torch.nn as nn
import math
import os
import json
import random
import sys
from utils.config import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

if args['data_name'] == 'multiwoz':
    data_path = 'data/newAgentUser/'
    args['data_name'] = 'newSpUser' # we use the name denoting the modified dataset
    from utils.utils_MultiWoz import prepare_combine_data
elif args['data_name'] == 'dailydialog':
    args['data_name'] = 'DailyUser'
    data_path = 'data/newDailyDialog/'
    from utils.utils_DailyDialog import prepare_combine_data
elif args['data_name'] == 'ccpe':
    args['data_name'] = 'CCPEAgent'
    data_path = 'data/newCCPE/'
    from utils.utils_CCPE import prepare_combine_data
else:
    logging.info('wrong input...')
    sys.exit(1)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)


# 其他参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_freq = 100  # 每隔print_freq个iteration打印状态
set_seed(args['seed'])

# 数据集 + 去标点 + 模型 + 参数
if args['punc']:
    punc_name = '_punc'
else:
    punc_name = ''
data_name = args['data_name'] + punc_name


def collate_fn(data):
    def _pad_seqs(seqs):
        lengths = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lengths)).long() # [batch_size, max_len] and PAD_token=0
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

def accuracy(pred, label):
    batch_size = label.size(0)
    corrects = (pred == label).sum()
    return corrects.item() * (100.0 / batch_size)

def random_gen(label):

    label_ratio = {}
    label_ratio['newAgent'] = [4302,5856]
    label_ratio['newUser'] = [7178,3123]
    label_ratio['newSpAgent'] = [8139,5346]
    label_ratio['newSpUser'] = [6983,6573]
    label_ratio['DailyAgent'] = [4770,2995]
    label_ratio['DailyUser'] = [3689, 4510]
    label_ratio['CCPEAgent'] = [894, 464]
    total = label_ratio[args['data_name']][0] + label_ratio[args['data_name']][1]
    total_vec = torch.randint_like(label, low=1, high=total)
    for i in range(len(total_vec)):
        total_vec[i] = 1 if total_vec[i] <= label_ratio[args['data_name']][0] else 0
    return total_vec

def testing(test_loader, print_freq, device):

    accs = AverageMeter()  # 一个batch的平均正确率

    for i, (_, _, _, label) in enumerate(test_loader):  

        label = label.to(device)

        pred = random_gen(label).to(device)
        # 计算准确率
        accs.update(accuracy(pred, label))
        
        if i % print_freq  == 0:
            print('Test: [{0}/{1}]\t'
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(test_loader),
                                                                            acc=accs))

    # 计算整个测试集上的正确率
    print('ACCURACY - {acc.avg:.3f}'.format(acc=accs))

    return accs.avg



def test(test_data):

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=args['batch_size'],
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              num_workers=4)

    acc = testing(test_loader=test_loader,
                              print_freq=print_freq,
                              device=device)
    return acc

if __name__ == "__main__":
    _, _, test_data  = prepare_combine_data(args['data_name'], args['nmt_name'], args['punc'])
    test(test_data)