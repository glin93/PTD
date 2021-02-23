import json
import random
import torch
import numpy as np
import pandas as pd
import re
import os
from collections import Counter

if __name__ == '__main__':
    from config import *
else:
    from utils.config import *




def genClassWordmap(data_name, min_word_freq=2):
    # 针对分类模型使用，没有加入特殊的词: '$u','$a','$$$$'

    data_path = 'data/newCCPE/' + data_name

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

    word2idx = {'PAD':PAD_token, 'UNK':UNK_token, 'SOS':SOS_token, 'EOS':EOS_token}
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word2idx.update({k: v + 4 for v, k in enumerate(words)}) # v+4是因为前面占了4个位置

    logging.info('vocab size: {0}'.format(len(word2idx)))

    with open(os.path.join(data_path, 'class_wordmap.json'), 'w') as f:
        json.dump(word2idx, f)
    
    logging.info('min_word_freq:{} | generate {} word map done'.format(min_word_freq, data_name))


def init_embeddings(embeddings):
    '''
    使用均匀分布U(-bias, bias)来随机初始化
    
    :param embeddings: 词向量矩阵
    '''
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, emb_format, word_map):
    '''
    加载预训练词向量
    
    :param emb_file: 词向量文件路径
    :param emb_format: 词向量格式: 'glove' or 'word2vec'
    :param word_map: 词表
    :return: 词向量矩阵, 词向量维度
    '''
    assert emb_format in {'glove', 'word2vec'}
    
    vocab = set(word_map.keys())
    
    logging.info("Loading embedding...")
    cnt = 0 # 记录读入的词数
    
    if emb_format == 'glove':
        
        with open(emb_file, 'r', encoding='utf-8') as f:
            emb_dim = len(f.readline().split(' ')) - 1 

        embeddings = torch.FloatTensor(len(vocab), emb_dim)
        #初始化词向量(对OOV进行随机初始化，即对那些在词表上的词但不在预训练词向量中的词)
        init_embeddings(embeddings)
        
        
        # 读入词向量文件
        for line in open(emb_file, 'r', encoding='utf-8'):
            line = line.split(' ')
            emb_word = line[0]

            # 过滤空值并转为float型
            embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

            # 如果不在词表上
            if emb_word not in vocab:
                continue
            else:
                cnt+=1

            embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

        logging.info("Number of words read: {0}".format(cnt))
        logging.info("Number of OOV: {0}".format(len(vocab)-cnt))

        return embeddings, emb_dim
    
    else:
        
        vectors = Vectors.load_word2vec_format(emb_file,binary=True)
        logging.info("Load successfully")
        emb_dim = 300
        embeddings = torch.FloatTensor(len(vocab), emb_dim)
        #初始化词向量(对OOV进行随机初始化，即对那些在词表上的词但不在预训练词向量中的词)
        init_embeddings(embeddings)
        
        for emb_word in vocab:
            
            if emb_word in vectors.index2word:
                
                embedding = vectors[emb_word]
                cnt += 1
                embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)
                
            else:          
                continue
            
        logging.info("Number of words read: ", cnt)
        logging.info("Number of OOV: ", len(vocab)-cnt)
        
        return embeddings, emb_dim


def process_embedding(data_name, word_dict):
    #这里word_dict是lang.word2idx

    data_path = 'data/newCCPE/' + data_name
    emb_file = 'data/glove.6B.300d.txt'
    emb_format = 'glove'
    pretrain_embed, embed_dim = load_embeddings(emb_file,  emb_format, word_dict)
    embed = dict()
    embed['pretrain'] = pretrain_embed
    embed['dim'] = embed_dim
    torch.save(embed, os.path.join(data_path, 'pretrain_embed.pt'))
    logging.info('process {} embed done.'.format(data_name))


class Lang:
    # 建立语言的字典
    def __init__(self, wordmap_json_path):
        with open(wordmap_json_path, 'r') as j:
            self.word2index = json.load(j)
        self.index2word = {v:k for k,v in self.word2index.items()}
        self.n_words = len(self.word2index)
    
    def __getitem__(self, key):
        if isinstance (key, int):
            return self.index2word[key]
        elif isinstance (key, str):
            if key not in self.word2index:
                return UNK_token
            else:
                return self.word2index[key]
    
    def __len__(self):
        return self.n_words

def read_langs(file_name, least_turn=1):
    # 读取并处理DailyAgent数据集
    def find_index(s):
        new_turn_cnt = len(s.split('|')) - 1 - least_turn
        cnt = 0
        for i in range(len(s)):
            if s[i] == '|':
                cnt+=1
            if cnt == new_turn_cnt:
                return i
        return 0

    with open(file_name, 'r') as j:
        raw_data = json.load(j)

    agentX, agentY = [], []
    userX, userY = [], []

    for dialog in raw_data:
        context_str = ''
        agent_x, agent_y = [], []
        user_x, user_y = [], []
        drop_dialog = False 
        for dia_turn in dialog['logs']:
            context_str += ' |' # 每轮开始的标识
            if context_str[-1] != ' ':
                context_str += ' '
            context_str += dia_turn['user']
            # 如果当前turn不是当前logs的第0个turn，那么当前句子肯定是user对上一句agent的回复
            if int(dia_turn['turn']) != 0:
                user_y.append(dia_turn['user'].strip())
                # logging.info('user_y true')
                    
            if len(dia_turn['agent']) == 0 and int(dia_turn['turn']) == len(dialog['logs'])-1: # agent为空，并且是当前logs的最后一个turn了
                continue
            
            if len(dia_turn['agent']) == 0 and int(dia_turn['turn']) != len(dialog['logs'])-1: # agent为空，但不是最后一个turn，这种就容易出错
                drop_dialog = True # 直接把整个dialog丢掉
                break
            
            for dia_subturn in dia_turn['agent']:
                # 如果本身agent就只有一句, 那么当前句子肯定是下一句user的历史
                # 注意：排除当前turn是最后一个turn的情况，因为此时没有下一句了
                if len(dia_turn['agent']) == 1 and int(dia_turn['turn']) != len(dialog['logs'])-1:
                    if context_str[-1] != ' ':
                        context_str += ' '
                    context_str += dia_subturn['text']
                    user_x.append(context_str[find_index(context_str):].replace('| ','').replace(' |','').strip())
                    # logging.info('user_x true')
                # 如果本身agent不只一句
                elif len(dia_turn['agent']) != 1:
                    # 如果当前句子不是agent的第0个句子，那么当前句子肯定是agent对上一句agent的回复
                    if int(dia_subturn['subturn']) != 0:
                        agent_y.append(dia_subturn['text'].strip())
                        # logging.info('agent_y true')
                    
                    # 如果当前句子不是agent最后一句，那么当前句子肯定是下一句agent的历史
                    if int(dia_subturn['subturn']) != len(dia_turn['agent']) - 1:
                        if context_str[-1] != ' ':
                            context_str += ' '
                        context_str += dia_subturn['text']
                        agent_x.append(context_str[find_index(context_str):].replace('| ','').replace(' |','').strip())
                        # logging.info('agent_x true')
                    
                    # 如果当前句子是agent最后一句，那么当前句子肯定是下一句user的历史
                    # 注意：排除当前turn是最后一个turn的情况，因为此时没有下一句了
                    if int(dia_subturn['subturn']) == len(dia_turn['agent']) - 1 and int(dia_turn['turn']) != len(dialog['logs'])-1:
                        if context_str[-1] != ' ':
                            context_str += ' '
                        context_str += dia_subturn['text']
                        user_x.append(context_str[find_index(context_str):].replace('| ','').replace(' |','').strip())
                        # logging.info('user_x true')
        
        if drop_dialog == False:
            agentX.extend(agent_x)
            agentY.extend(agent_y)
            userX.extend(user_x)
            userY.extend(user_y)
        assert len(agentX) == len(agentY)
        assert len(userX) == len(userY)

    return agentX, agentY, userX, userY


def read_langs_USER(file_name, least_turn=1):
    # 读取并处理DailyUser数据
    def find_index(s):
        new_turn_cnt = len(s.split('|')) - 1 - least_turn
        cnt = 0
        for i in range(len(s)):
            if s[i] == '|':
                cnt+=1
            if cnt == new_turn_cnt:
                return i
        return 0

    with open(file_name, 'r') as j:
        raw_data = json.load(j)

    agentX, agentY = [], []
    userX, userY = [], []


    for dialog in raw_data:
        context_str = ''

        agent_x, agent_y = [], [] # 表示agent回复
        user_x, user_y = [], [] # 表示user还会继续说

        for dia_turn in dialog['logs']:
            context_str += ' |' # 每轮开始的标识
            if context_str[-1] != ' ':
                context_str += ' '
            # 如果当前轮user为空，则跳过该turn
            if len(dia_turn['user']) == 0:
                context_str += dia_turn['agent']
                continue
            for dia_subturn in dia_turn['user']:
                # 如果当前轮user没有被拆分，那么当前句子一定是下一句agent的历史，即agent_x
                if(len(dia_turn['user']) == 1):
                    if context_str[-1] != ' ':
                        context_str += ' '
                    context_str += dia_subturn['text']
                    # 注意，排除没有下一句agent的情况
                    if 'agent' in dia_turn.keys():
                        agent_x.append(context_str[find_index(context_str):].replace('| ','').replace(' |','').strip())
                    continue
                # 如果当前user是最后一句user，那么当前句子一定是下一句agent的历史，同时是上一句user的接下来说的
                
                if(int(dia_subturn['subturn']) == len(dia_turn['user']) - 1):
                    if context_str[-1] != ' ':
                        context_str += ' '
                    context_str += dia_subturn['text']
                    # 注意，排除没有下一句agent的情况
                    if 'agent' in dia_turn.keys():
                        agent_x.append(context_str[find_index(context_str):].replace('| ','').replace(' |','').strip())
                    user_y.append(dia_subturn['text'].strip())
                # 如果当前user是第一句，且当前轮不止一句的话，那么当前句子一定是下一句user的历史
                elif (int(dia_subturn['subturn']) == 0):
                    if context_str[-1] != ' ':
                        context_str += ' '
                    context_str += dia_subturn['text']
                    user_x.append(context_str[find_index(context_str):].replace('| ','').replace(' |','').strip())
                # 如果当前user不是最后一句，那么当前句子是下一句user的历史，也是上一句user的接下来说的
                else:
                    if context_str[-1] != ' ':
                        context_str += ' '
                    context_str += dia_subturn['text']
                    user_x.append(context_str[find_index(context_str):].replace('| ','').replace(' |','').strip())
                    user_y.append(dia_subturn['text'].strip())
            # agent也要加到历史里面
            if context_str[-1] != ' ':
                context_str += ' '
            if 'agent' in dia_turn.keys():
                context_str += dia_turn['agent']
                # user弄完以后，接下来的肯定是一句agent，当前句子一定是上一句user的回复，配对起来就是agent_x, agent_y
                # 前提是有这个agent
                agent_y.append(dia_turn['agent'].strip())

        agentX.extend(agent_x)
        agentY.extend(agent_y)
        userX.extend(user_x)
        userY.extend(user_y)
        assert len(agentX) == len(agentY)
        assert len(userX) == len(userY)
   
    
    # agent: 指的是回复
    # user: 指的是下一句接着说

    return agentX, agentY, userX, userY

def write_txt(split, obj, src_data, trg_data, data_name):
    # 将经过read langs处理好的数据写入txt，方便openNMT调用
    assert len(src_data) == len(trg_data)

    with open('data/newCCPE/' + data_name + '/src-' + split + '-' + obj + '.txt', 'w') as f:
        for src in src_data:
            f.write(src + '\n')
    with open('data/newCCPE/' + data_name + '/tgt-' + split + '-' + obj + '.txt', 'w') as f:
        for trg in trg_data:
            f.write(trg + '\n')


def process_NMT(data_name):
    # 将经过read langs处理好的数据写入txt，方便openNMT调用和后续使用

    data_path = 'data/newCCPE/' + data_name + '/' + data_name
    if data_name[-1] == 't':
        train_agentX, train_agentY, train_userX, train_userY = read_langs(data_path + 'Train_dials.json')
        val_agentX, val_agentY, val_userX, val_userY = read_langs(data_path + 'Val_dials.json')
        test_agentX, test_agentY, test_userX, test_userY = read_langs(data_path + 'Test_dials.json')
    else:
        train_agentX, train_agentY, train_userX, train_userY = read_langs_USER(data_path + 'Train_dials.json')
        val_agentX, val_agentY, val_userX, val_userY = read_langs_USER(data_path + 'Val_dials.json')
        test_agentX, test_agentY, test_userX, test_userY = read_langs_USER(data_path + 'Test_dials.json')
    
    logging.info('{}--TRAIN AGENT LEN: {} | VAL AGENT LEN: {} | TEST AGENT LEN {}'.format(data_name, len(train_agentX), len(val_agentX), len(test_agentX)))
    logging.info('{}--TRAIN USER LEN: {} | VAL USER LEN: {} | TEST USER LEN {}'.format(data_name, len(train_userX), len(val_userX), len(test_userX)))

    write_txt('train', 'agent', train_agentX, train_agentY, data_name)
    write_txt('val', 'agent', val_agentX, val_agentY, data_name)
    write_txt('test', 'agent', test_agentX, test_agentY, data_name)

    write_txt('train', 'user', train_userX, train_userY, data_name)
    write_txt('val', 'user', val_userX, val_userY, data_name)
    write_txt('test', 'user', test_userX, test_userY, data_name)
    logging.info('write {} NMT sucess'.format(data_name))





def get_combine_data(split, wordmap, data_name, nmt_name, preserve_punctuation=True):

    if preserve_punctuation:
        data_path = 'data/newCCPE/' + data_name + '/combine_punc_' + nmt_name + '_' + split + '.json'
    else:
        data_path = 'data/newCCPE/' + data_name + '/combine_' + nmt_name + '_' + split + '.json'

    if os.path.exists(data_path):
        return False

    # 输入是his，nxt，res, 用于trainThreeCNN
    # 这里的nmt name主要为了后面改变nmt参数时候用, 不同nmt，预测结果文件不同

    agent_his = []; user_his = []
    agent_for_agent = []; agent_for_user = []
    user_for_agent = []; user_for_user = []

    with open('data/newCCPE/' + data_name + '/src-' + split + '-agent.txt', 'r') as f:
        for line in f.readlines():
            agent_his.append(line.strip())
    with open('data/newCCPE/' + data_name + '/src-' + split + '-user.txt', 'r') as f:
        for line in f.readlines():
            user_his.append(line.strip())
    with open('data/newCCPE/' + data_name + '/pred-' + split + '-agent-for-agent-' + nmt_name + '.txt', 'r') as f:
        # 用agent的生成器去处理agent数据集的结果
        for line in f.readlines():
            agent_for_agent.append(line.strip())
    with open('data/newCCPE/' + data_name + '/pred-' + split + '-agent-for-user-' + nmt_name + '.txt', 'r') as f:
        # 用agent的生成器去处理user数据集的结果
        for line in f.readlines():
            agent_for_user.append(line.strip())
    with open('data/newCCPE/' + data_name + '/pred-' + split + '-user-for-agent-' + nmt_name + '.txt', 'r') as f:
        # 用user的生成器去处理agent数据集的结果
        for line in f.readlines():
            user_for_agent.append(line.strip())
    with open('data/newCCPE/' + data_name + '/pred-' + split + '-user-for-user-' +  nmt_name + '.txt', 'r') as f:
        # 用user的生成器去处理user数据集的结果
        for line in f.readlines():
            user_for_user.append(line.strip())
    data = []
    for his, nxt, res in zip(agent_his, agent_for_agent, user_for_agent):
        # 用agent的生成器和user生成器去处理agent的his
        if preserve_punctuation == False:
            his = his.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
            nxt = nxt.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
            res = res.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
        his_token = [wordmap[word] for word in his.split()]  + [EOS_token]
        nxt_token = [wordmap[word] for word in nxt.split()]  + [EOS_token]
        res_token = [wordmap[word] for word in res.split()]  + [EOS_token]
        label = 1
        data.append({'his':his_token, 'nxt':nxt_token, 'res':res_token, 'label':label})
    for his, nxt, res in zip(user_his, agent_for_user, user_for_user):
        # 用agent的生成器和user生成器去处理agent的his
        if preserve_punctuation == False:
            his = his.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
            nxt = nxt.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
            res = res.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
        his_token = [wordmap[word] for word in his.split()]  + [EOS_token]
        nxt_token = [wordmap[word] for word in nxt.split()]  + [EOS_token]
        res_token = [wordmap[word] for word in res.split()]  + [EOS_token]
        label = 0
        data.append({'his':his_token, 'nxt':nxt_token, 'res':res_token, 'label':label})
    random.shuffle(data)
    if preserve_punctuation:
        with open('data/newCCPE/' + data_name + '/combine_punc_' + nmt_name + '_' + split + '.json', 'w') as j:
            json.dump(data, j)
    else:
        with open('data/newCCPE/' + data_name + '/combine_' + nmt_name + '_' + split + '.json', 'w') as j:
            json.dump(data, j)
    
    return True


class CombineDataset(torch.utils.data.Dataset):
    def __init__(self, split, data_name, nmt_name, preserve_punctuation):
        if preserve_punctuation:
            with open('data/newCCPE/' + data_name + '/combine_punc_' + nmt_name + '_' + split + '.json', 'r') as j:
                self.data = json.load(j)
                self.len_data = len(self.data)
        else:
            with open('data/newCCPE/' + data_name + '/combine_' + nmt_name + '_' + split + '.json', 'r') as j:
                self.data = json.load(j)
                self.len_data = len(self.data)            


    def __len__(self):
        return self.len_data
        
    def __getitem__(self, index):
        his = torch.tensor(self.data[index]['his'])
        nxt = torch.tensor(self.data[index]['nxt'])
        res = torch.tensor(self.data[index]['res'])
        label = self.data[index]['label']
        return his, nxt, res, label

def prepare_combine_data(data_name, nmt_name, preserve_punctuation):

    train_data = CombineDataset('train', data_name, nmt_name, preserve_punctuation)
    val_data = CombineDataset('val', data_name, nmt_name, preserve_punctuation)
    test_data = CombineDataset('test', data_name, nmt_name, preserve_punctuation)
    return train_data, val_data, test_data



def write_tsv_for_bert(split, data_name, task, preserve_punctuation, nmt_name=None):

    assert task in {'his', 'combine'}
    assert nmt_name in {None, 'origin', 'glove', 'woattn'}
    
    agent_his = []; user_his = []
    agent_for_agent = []; agent_for_user = []
    user_for_agent = []; user_for_user = []
    data = []
    if task == 'his':
        if preserve_punctuation:
            data_path = 'data/newCCPE/' + data_name + '/his_punc_' + split + '.tsv'
        else:
            data_path = 'data/newCCPE/' + data_name + '/his_' + split + '.tsv'

        if os.path.exists(data_path):
            return False

        with open('data/newCCPE/' + data_name + '/src-' + split + '-agent.txt', 'r') as f:
            for line in f.readlines():
                his = line.strip()
                if preserve_punctuation == False:
                    his = his.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                data.append({'his': his, 'label': 1})
        with open('data/newCCPE/' + data_name + '/src-' + split + '-user.txt', 'r') as f:
            for line in f.readlines():
                his = line.strip()
                if preserve_punctuation == False:
                    his = his.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                data.append({'his': his, 'label': 0})
        random.shuffle(data)
        df = pd.DataFrame(data)
        if preserve_punctuation:
            df.to_csv('data/newCCPE/' + data_name + '/his_punc_' + split + '.tsv', index=False, sep='\t')
        else:
            df.to_csv('data/newCCPE/' + data_name + '/his_' + split + '.tsv', index=False, sep='\t')
        logging.info('BERT {} DATA LEN: {} | punc={} '.format(split, len(df), preserve_punctuation))

        return True

    elif task == 'combine':

        with open('data/newCCPE/' + data_name + '/src-' + split + '-agent.txt', 'r') as f:
            for line in f.readlines():
                agent_his.append(line.strip())
        with open('data/newCCPE/' + data_name + '/src-' + split + '-user.txt', 'r') as f:
            for line in f.readlines():
                user_his.append(line.strip())
        with open('data/newCCPE/' + data_name + '/pred-' + split + '-agent-for-agent-' + nmt_name + '.txt', 'r') as f:
            # 用agent的生成器去处理agent数据集的结果
            for line in f.readlines():
                agent_for_agent.append(line.strip())
        with open('data/newCCPE/' + data_name + '/pred-' + split + '-agent-for-user-' + nmt_name + '.txt', 'r') as f:
            # 用agent的生成器去处理user数据集的结果
            for line in f.readlines():
                agent_for_user.append(line.strip())
        with open('data/newCCPE/' + data_name + '/pred-' + split + '-user-for-agent-' + nmt_name + '.txt', 'r') as f:
            # 用user的生成器去处理agent数据集的结果
            for line in f.readlines():
                user_for_agent.append(line.strip())
        with open('data/newCCPE/' + data_name + '/pred-' + split + '-user-for-user-' +  nmt_name + '.txt', 'r') as f:
            # 用user的生成器去处理user数据集的结果
            for line in f.readlines():
                user_for_user.append(line.strip())      
        

        if args['addition_name'] == 'directClassify':
            if preserve_punctuation:
                data_path = 'data/newCCPE/' + data_name + '/direct_punc_' + split + '.tsv'
            else:
                data_path = 'data/newCCPE/' + data_name + '/direct_' + split + '.tsv'
            if os.path.exists(data_path):
                return False

            data = []
            for his, nxt, res in zip(agent_his, agent_for_agent, user_for_agent):
                # 用agent的生成器和user生成器去处理agent的his, label=1, 表示选择接下来继续说
                if preserve_punctuation == False:
                    his = his.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    nxt = nxt.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    res = res.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                all_text = his + '[SEP]' + nxt + '[SEP]' + res
                label = 1
                data.append({'combine': all_text, 'label': label})
            for his, nxt, res in zip(user_his, agent_for_user, user_for_user):
                # 用agent的生成器和user生成器去处理user的his, label=0， 表示选择回复
                if preserve_punctuation == False:
                    his = his.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    nxt = nxt.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    res = res.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                all_text = his + '[SEP]' + nxt + '[SEP]' + res
                label = 0 
                data.append({'combine': all_text, 'label': label})
            random.shuffle(data)
            df = pd.DataFrame(data)
            if preserve_punctuation:
                df.to_csv('data/newCCPE/' + data_name + '/direct_punc_' + split + '.tsv', index=False, sep='\t')
            else:
                df.to_csv('data/newCCPE/' + data_name + '/direct_' + split + '.tsv', index=False, sep='\t')
            logging.info('BERT {} DATA LEN: {} | punc={} '.format(split, len(df), preserve_punctuation))

            return True   

        else:

            if preserve_punctuation:
                data_path = 'data/newCCPE/' + data_name + '/combine_punc_' + split + '.tsv'
            else:
                data_path = 'data/newCCPE/' + data_name + '/combine_' + split + '.tsv'

            if os.path.exists(data_path):
                return False

            data = []
            for his, nxt, res in zip(agent_his, agent_for_agent, user_for_agent):
                # 用agent的生成器和user生成器去处理agent的his, label=1, 表示选择接下来继续说
                sents = re.split('(\,|\!|\.|\?)', his)
                l = []
                for j in range(int(len(sents)/2)):
                    sent = sents[2*j] + sents[2*j+1]
                    l.append(sent)
                if len(l) > 1:
                    context = "".join(l[:-1])
                    question = l[-1].strip()
                elif len(l) == 1:
                    context = "[PAD]"
                    question = l[0]
                else:
                    context = "[PAD]"
                    question = sents[0]
                answer1 = nxt
                answer0 = res
                label = 1
                if preserve_punctuation == False:
                    context = context.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    question = question.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    answer1 = answer1.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    answer0 = answer0.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                data.append({'context': context, 'question':question, 'answer0': answer0, 'answer1':answer1, 'label': label})
            for his, nxt, res in zip(user_his, agent_for_user, user_for_user):
                # 用agent的生成器和user生成器去处理user的his, label=0， 表示选择回复
                sents = re.split('(\,|\!|\.|\?)', his)
                l = []
                for j in range(int(len(sents)/2)):
                    sent = sents[2*j] + sents[2*j+1]
                    l.append(sent)
                if len(l) > 1:
                    context = "".join(l[:-1])
                    question = l[-1].strip()
                elif len(l) == 1:
                    context = "[PAD]"
                    question = l[0]
                else:
                    context = "[PAD]"
                    question = sents[0]
                answer1 = nxt
                answer0 = res
                label = 0
                if preserve_punctuation == False:
                    context = context.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    question = question.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    answer1 = answer1.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                    answer0 = answer0.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@\\^_`{|}~'))
                data.append({'context': context, 'question':question, 'answer0': answer0, 'answer1':answer1, 'label': label})
            
            random.shuffle(data)
            df = pd.DataFrame(data)
            df['id'] = range(len(df))
            if preserve_punctuation:
                df.to_csv('data/newCCPE/' + data_name + '/combine_punc_' + split + '.csv', index=False)
            else:
                df.to_csv('data/newCCPE/' + data_name + '/combine_' + split + '.csv', index=False)
            logging.info('BERT {} DATA LEN: {} | punc={} '.format(split, len(df), preserve_punctuation))

            return True
        


if __name__ == "__main__":
    wordmap_path = os.path.join('data/newCCPE', 'CCPEAgent', 'class_wordmap.json')
    embed_path = os.path.join('data/newCCPE', 'CCPEAgent', 'pretrain_embed.pt')

    if os.path.exists(wordmap_path) == False:
        genClassWordmap('CCPEAgent', min_word_freq=0)

    lang = Lang(wordmap_path)
    if os.path.exists(embed_path) == False:
        process_embedding('CCPEAgent', lang.word2index)

    for split in ['train', 'val', 'test']:
        for obj in ['user', 'agent']:
            data_path = 'data/newCCPE/' + 'CCPEAgent'
            src_path = os.path.join(data_path, 'src-' + split + '-' + obj + '.txt')
            tgt_path = os.path.join(data_path, 'tgt-' + split + '-' + obj + '.txt')
            if os.path.exists(src_path) == False or os.path.exists(tgt_path) == False:
                process_NMT('CCPEAgent')

    if args['combine']:
        get_combine_data('train', lang, 'CCPEAgent', args['nmt_name'], False)
        get_combine_data('val', lang, 'CCPEAgent', args['nmt_name'], False)
        get_combine_data('test', lang, 'CCPEAgent', args['nmt_name'], False)

        get_combine_data('train', lang, 'CCPEAgent', args['nmt_name'], True)
        get_combine_data('val', lang, 'CCPEAgent', args['nmt_name'], True)
        get_combine_data('test', lang, 'CCPEAgent', args['nmt_name'], True)
    
    if args['process_bert']:
        write_tsv_for_bert('train', 'CCPEAgent', 'his', False)
        write_tsv_for_bert('val', 'CCPEAgent', 'his', False)
        write_tsv_for_bert('test', 'CCPEAgent', 'his', False)

        write_tsv_for_bert('train', 'CCPEAgent', 'his', True)
        write_tsv_for_bert('val', 'CCPEAgent', 'his', True)
        write_tsv_for_bert('test', 'CCPEAgent', 'his', True)

        write_tsv_for_bert('train', 'CCPEAgent', 'combine', False, args['nmt_name'])
        write_tsv_for_bert('val', 'CCPEAgent', 'combine', False, args['nmt_name'])
        write_tsv_for_bert('test', 'CCPEAgent', 'combine', False, args['nmt_name'])

        write_tsv_for_bert('train', 'CCPEAgent', 'combine', True, args['nmt_name'])
        write_tsv_for_bert('val', 'CCPEAgent', 'combine', True, args['nmt_name'])
        write_tsv_for_bert('test', 'CCPEAgent', 'combine', True, args['nmt_name'])
