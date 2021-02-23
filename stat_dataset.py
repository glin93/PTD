# 数据集有多少个turn，多少段对话，平均turn，平均words，user平均每个turn有多少分句之类的

import json
from collections import Counter
import os

def read_langs(file_name):
    count = {'dialog_num':0, 'total_turn_num':0, 'total_subturn_num':0} 
    with open(file_name, 'r') as j:
        raw_data = json.load(j)
    for dialog in raw_data:
        drop_dialog = False 
        for dia_turn in dialog['logs']:     
            if len(dia_turn['agent']) == 0 and int(dia_turn['turn']) != len(dialog['logs'])-1: # agent为空，但不是最后一个turn，这种就容易出错
                drop_dialog = True # 直接把整个dialog丢掉
                break
            count['total_turn_num'] += 1
            for _ in dia_turn['agent']:
                count['total_subturn_num'] += 1  
        if drop_dialog == False:
            count['dialog_num'] += 1

    count['avg_turn_per_dialog'] = count['total_turn_num'] / count['dialog_num']
    count['avg_subturn_per_turn'] = count['total_subturn_num'] / count['total_turn_num']

    return count

def read_langs_USER(file_name):
    count = {'dialog_num':0, 'total_turn_num':0, 'total_subturn_num':0} 
    with open(file_name, 'r') as j:
        raw_data = json.load(j)
    for dialog in raw_data:
        for dia_turn in dialog['logs']:
            # 如果当前轮user为空，则跳过该turn
            if len(dia_turn['user']) == 0:
                continue
            count['total_turn_num'] += 1
            for _ in dia_turn['user']:
                # 如果当前轮user没有被拆分，那么当前句子一定是下一句agent的历史，即agent_x
                count['total_subturn_num'] += 1
        count['dialog_num'] += 1
    # agent: 指的是回复
    # user: 指的是下一句接着说

    count['avg_turn_per_dialog'] = count['total_turn_num'] / count['dialog_num']
    count['avg_subturn_per_turn'] = count['total_subturn_num'] / count['total_turn_num']

    return count


def word_count(data_name, dataset_name):
    # 针对分类模型使用，没有加入特殊的词: '$u','$a','$$$$'

    data_path = 'data/' + dataset_name + '/' + data_name

    with open(os.path.join(data_path, data_name + 'Test_dials.json'), 'r') as f:
        test = json.load(f)
    with open(os.path.join(data_path,  data_name + 'Train_dials.json'), 'r') as f:
        train = json.load(f)
    with open(os.path.join(data_path, data_name + 'Val_dials.json'), 'r') as f:
        val = json.load(f)
    
    word_freq = Counter()
    def word_gen(data, data_name):
        if data_name[-1] == 't':
            for item in data:
                for turn in item['logs']:         
                    word_freq.update(turn['user'].split())
                    for subturn in turn['agent']:
                        word_freq.update(subturn['text'].split())
        else:
            for item in data:
                for turn in item['logs']:
                    if 'agent' in turn.keys():
                        word_freq.update(turn['agent'].split())
                    for subturn in turn['user']:
                        word_freq.update(subturn['text'].split())
    word_gen(test, data_name)
    word_gen(train, data_name)
    word_gen(val, data_name)

    words = [w for w in word_freq.keys()]
    print(dataset_name, data_name, len(words))



def read_txt(split, data_name, dataset):
    # obj: agent, user, 针对USER数据集，agent: 指的是回复，user: 指的是下一句接着说
    # 无论什么数据集，agent label=1， user label=0
    # 比如USER数据集：obj=agent，表示agent回复user，his是user说的，label=1；obj=user，表示user会接下来说，label=0，his是user说的
    # 比如Agent数据集：obj=agent，表示agent接下来继续说，his是agent说的，label=1；obj=user，表示user会回复agent说的，label=0，his是agent说的
    # 即agent_his 表示 agent回复了上一句历史，对于USER数据集，his是user说的，对于Agent数据集，his是agent说的，label=1
    # user_his 表示 user回复了上一句历史，label=0
    # label表明的是当前这句该谁说，如果agent说，就是label=1；如果user说，label=0
    
    count = {'total_his_words':0, 'total_his_num':0, 'agent_label':0, 'user_label':0, 'total_response_words':0, 'total_response_num':0, 'agent_response_words':0, 'agent_response_num':0, 'user_response_words':0, 'user_response_num':0}

    for obj in {'agent', 'user'}:
        with open('data/' + dataset + '/' + data_name + '/src-' + split + '-' + obj + '.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                count['total_his_words'] += len(line.split())
                count['total_his_num'] += 1
                if obj == 'agent':
                    count['agent_label'] += 1
                else:
                    count['user_label'] += 1
        with open('data/' + dataset + '/' + data_name + '/tgt-' + split + '-' + obj + '.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                count['total_response_words'] += len(line.split())
                count['total_response_num'] += 1
                if obj == 'agent':
                    #实际针对USER数据集，agent指的是agent回复user说的，user指的user回复上一句user
                    #针对Agent数据集，agent指的是agent回复上一句agent说的，user指的user回复agent说的
                    count['agent_response_words'] += len(line.split()) 
                    count['agent_response_num'] += 1
                else:
                    count['user_response_words'] += len(line.split()) 
                    count['user_response_num'] += 1                 

    count['avg_his_words'] = count['total_his_words'] / count['total_his_num']
    count['avg_response_words'] = count['total_response_words'] / count['total_response_num']
    count['avg_agent_response_words'] = count['agent_response_words'] / count['agent_response_num']
    count['avg_user_response_words'] = count['user_response_words'] / count['user_response_num']
    count['label_ratio'] = "{}:{}={}".format(count['agent_label'], count['user_label'], count['agent_label']/count['user_label'])
    count['data_len'] = count['agent_label'] + count['user_label']

    return count


if __name__ == "__main__":

    results = {}

    for dataset in {'newAgentUser', 'newDailyDialog', 'newCCPE'}:
        results[dataset] = {}
        if dataset == 'newAgentUser':
            data_names = {'newSpUser'}
        elif dataset == 'newDailyDialog':
            data_names = {'DailyUser'}
        else:
            data_names = {'CCPEAgent'}
        for data_name in data_names:
            results[dataset][data_name] = {}
            for split in {'train','val','test'}:
                if split == 'train':
                    data_path = os.path.join('data', dataset, data_name, data_name + 'Train_dials.json')
                elif split == 'val':
                    data_path = os.path.join('data', dataset, data_name, data_name + 'Val_dials.json')
                else:
                    data_path = os.path.join('data', dataset, data_name, data_name + 'Test_dials.json')
                
                results[dataset][data_name][split] = {}
                if data_name[-1] == 't':
                    results[dataset][data_name][split].update(read_langs(data_path))
                else:
                    results[dataset][data_name][split].update(read_langs_USER(data_path))
                results[dataset][data_name][split].update(read_txt(split, data_name, dataset))

            with open(os.path.join('data', dataset, data_name, 'class_wordmap.json'), 'r') as f:
                results[dataset][data_name]['vocab_size'] = len(json.load(f))

            results[dataset][data_name]['data_ratio'] = "Train:Val:Test={}:{}:{}".format(results[dataset][data_name]['train']['data_len'], results[dataset][data_name]['val']['data_len'], results[dataset][data_name]['test']['data_len'])

    with open(os.path.join('output', 'stat_res.json'), 'w') as f:
        json.dump(results, f)



