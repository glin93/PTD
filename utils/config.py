import argparse
import logging


PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S')
parser = argparse.ArgumentParser(description='One to Many')
parser.add_argument('--seed', required=False, default=520, type=int)
parser.add_argument('--data_name', required=False, default='CCPE', type=str)
parser.add_argument('--process_bert', required=False, default=False, action='store_true')
parser.add_argument('--nmt_name', required=False, default='origin', type=str)
parser.add_argument('--combine', required=False, default=False, action='store_true')

parser.add_argument('--model_seed', required=False, default=520, type=int)
parser.add_argument('-hdim', '--hidden_size', required=False, default=300, type=int)
parser.add_argument('-birnn','--birnn', required=False, default=True, action='store_true')
parser.add_argument('-bsz','--batch_size', help='Batch_size', required=False, default=32, type=int)
parser.add_argument('--lr', required=False, default=1e-4, type=float)
parser.add_argument('-kn', '--kernel_num', required=False, default=100, type=int)
parser.add_argument('-ksz', '--kernel_sizes', required=False,  action='store', type=int, nargs='*', default=[3,4,5])
parser.add_argument('-dr', '--dropout', required=False, default=0.5,  type=float)
parser.add_argument('-st','--static', required=False, default=True, action='store_true')
parser.add_argument('-nst','--non_static', required=False, default=True, action='store_true')
parser.add_argument('-mc','--multichannel', required=False, default=True, action='store_true')
parser.add_argument('--punc', required=False, default=False, action='store_true')
parser.add_argument('-wd', '--weight_decay', required=False, default=1e-5, type=float)
parser.add_argument('-de', '--decay_epoch', required=False, default=10, type=int)
parser.add_argument('-ie', '--improvement_epoch', required=False, default=6, type=int)
parser.add_argument('-edd','--embed_dim_default', required=False, default=128,  type=int)
parser.add_argument('--epochs', required=False, default=120, type=int)
parser.add_argument('--single_mode', required=False, default=False, action='store_true')
parser.add_argument('--do_eval', required=False, default=False, action='store_true')
parser.add_argument('--do_train', required=False, default=False, action='store_true')
parser.add_argument('--addition_name', required=False, default='', type=str)
parser.add_argument('--ckpt', required=False, default='', type=str)
parser.add_argument('--do_sample', required=False, default=False, action='store_true')



args = vars(parser.parse_args())