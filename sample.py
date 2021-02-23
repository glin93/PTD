import random
import os
import pandas as pd
import time
from utils.config import *
sample_size = 20

if args['data_name'] == 'multiwoz':
    args['data_name'] = 'newSpUser' # we use the name denoting the modified dataset
elif args['data_name'] == 'dailydialog':
    args['data_name'] = 'DailyUser'
elif args['data_name'] == 'ccpe':
    args['data_name'] = 'CCPEAgent'
else:
    logging.info('wrong input...')
    sys.exit(1)

def sample_csv(data_name):
    baseline = pd.read_csv(os.path.join('output','Baseline_'+ data_name +'_textcnn_sample.csv'))
    our = pd.read_csv(os.path.join('output', data_name +'_textcnn_sample.csv'))
    df = pd.merge(baseline[['his', 'agent_gen', 'user_gen', 'label', 'correct_flag']], our[['his', 'correct_flag']], on='his')
    df.to_csv('output/tmp.csv')
    sample_data= df[(df['correct_flag_x'] == False) & (df['correct_flag_y'] == True)]
    sample_idx = random.sample(range(len(sample_data)), sample_size)
    out = sample_data.iloc[sample_idx]
    timestamp = time.strftime("%m-%d-%H:%M:%S", time.localtime()) 
    out.to_csv(os.path.join('output','result_'+ timestamp + '_' + data_name +'_textcnn_sample.csv'))

if __name__ == "__main__":
    sample_csv(args['data_name'])

