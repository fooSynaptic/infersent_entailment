# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import os
import numpy as np
import torch
import sys
import io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import time
import codecs


def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 512))
        #batch[:] = [b.remove(' ') for b in batch]
    for i in range(len(batch)):
        k, cnt, j = 0, 0, 0
        while j < len(batch[i]):
            j_word = ''.join(batch[i][k:j+1])
            if j_word in word_vec.keys():
                embed[cnt,i,:] =word_vec[j_word]
                cnt+=1                
            j=j+1
            k=j
        if cnt>0:
            lengths[i] = cnt
    return torch.from_numpy(embed).float(), lengths
'''
k = 0
cnt = 0
j = 0
while j < len(batch[i]):
  if batch[i][k:j+1] in word_vec:
    embed[cnt, i, :] = word_vec[batch[i][k:j+1]]
    cnt += 1
    k = j
  j += 1
'''
#    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        sent = sent.replace(' ','')
        for word in set(sent):
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    word_dict['。'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with codecs.open(glove_path, 'r', 'utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
            if word in ['？', '。',' ']:
                word_vec[word] = np.array(list(map(float, vec.split())))
            #try:
            #    word_vec[' ']
            #except:
            #    word_vec[' '] = np.array(list(map(float, '0'*len(vec))))
    print('Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict)))
    word_vec['<s>'] = word_vec['。']
    #word_vec[' '] = word_vec['回澜镇']
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec




#we want to make little transform of the function of get_nli
def get_nli(data_path): #return labeled datasets! this function read the labeled data and feed them into the rnn trainning model!
    s1 = {}
    s2 = {}
    target = {}
    #dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
    dico_label = {'entailment': 0,  'contradiction': 1}
    #for data_type in ['train', 'dev', 'test']:
    for data_type in ['train']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path,
                                                 'labels.' + data_type)
        with codecs.open(s1[data_type]['path'], 'r','utf-8') as f1:
            s1[data_type]['sent'] = [line.rstrip() for line in f1]
            print(len(s1[data_type]['sent']))
            sent_count = len(s1[data_type]['sent'])
            print(sent_count)
        with codecs.open(s2[data_type]['path'], 'r','utf-8') as f2:
            s2[data_type]['sent'] = [line.rstrip() for line in f2]
        try:
            target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]  #return a vector contain three kind of int {0,1,2}
                for line in codecs.open(target[data_type]['path'], 'r','utf-8')])
        except KeyError:
            print('label file contain some symbals to retrieve the dico_label')
        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
            len(target[data_type]['data'])
        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), sent_count, data_type))
    filename = './dataset/sorted_ques.txt'
    with open(filename,'rb') as f:
        raw_dim_ques = f.readlines()
        dim_ques = [dim.decode().strip('\n').split('||') for dim in raw_dim_ques]
        total_ques = []
        [total_ques.extend(_dim) for _dim in dim_ques]
        ques_to_dim = {}
        for i,dim in enumerate(dim_ques):
            for q in dim:
                ques_to_dim.setdefault(q,i)
    s1['train']['sent'][:] = [sent.replace(' ','') for sent in s1['train']['sent']]
    s2['train']['sent'][:] = [sent.replace(' ','') for sent in s2['train']['sent']]
    assert len(s1['train']['sent']) == len(s2['train']['sent']) == len(target['train']['data'])
    l = len(s1['train']['sent'])
    train = {'s1': s1['train']['sent'][:int(0.8*l)], 's2': s2['train']['sent'][:int(0.8*l)],'label': target['train']['data'][:int(0.8*l)]}
    dev = {'s1': s1['train']['sent'][int(0.8*l):int(0.9*l)], 's2': s2['train']['sent'][int(0.8*l):int(0.9*l)],'label': target['train']['data'][int(0.8*l):int(0.9*l)]}
    test = {'s1': s1['train']['sent'][int(0.9*l):], 's2': s2['train']['sent'][int(0.9*l):],'label': target['train']['data'][int(0.9*l):]}
    probe_seed = random.randrange(0,10,2)
    def _assert_dataset(data,seed):
        label = (True if str(data['label'][seed]) == '0' else False)
        q1 = data['s1'][seed]
        q2 = data['s2'][seed]
        if ques_to_dim[q1] == ques_to_dim[q2]:
            new = True
        else:
            new = False
        return new == label
    assert _assert_dataset(train,probe_seed)
    assert _assert_dataset(dev,probe_seed)
    assert _assert_dataset(test,probe_seed)
    return train, dev, test