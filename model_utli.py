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
from data import get_nli, build_vocab, get_batch
import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
#import faiss

'''
def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))
        #batch[:] = [b.remove(' ') for b in batch]
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            #print(batch[i][j])
            embed[j, i, :] = word_vec[batch[i][j]]
    return torch.from_numpy(embed).float(), lengths
'''

GLOVE_PATH = '<glove>/<path>'

wenda_infersent = torch.load('./glove_modeldir/GloVe.pickle')
wenda_infersent.encoder.enc_lstm.flatten_parameters()

train, valid, test = get_nli('./<corpus>/<path>')

train['s1'] = list(set(train['s1']))
train['s2'] = list(set(train['s2']))
print(len(train['s1']))

word_vec = build_vocab(train['s1'], GLOVE_PATH)

for split in ['s1', 's2']:
    for data_type in ['train']:
        eval(data_type)[split] = np.array([
            [word for word in list(sent) if word in word_vec] for sent in eval(data_type)[split]])

permutation = np.random.permutation(len(train['s1']))

s1 = train['s1'][permutation]
#word_vec = build_vocab(s1, GLOVE_PATH)

print([''.join(sent) for sent in s1[:50]])

wenda_cod = wenda_infersent.encoder



def wenda_query():
    print("tell me what kind of issue you want to consultate me?")
    q = input()
    #q='泰瑞莎'
    ps = time.time()
    q_set = [q]
    q_set.append(['。'])
    s1_batch, s1_len = get_batch(q_set,word_vec)
    result_tensor = []
    for idx in range(len(s1)):
        #print(idx)
        s2_set = [s1[idx]]
        s2_set.append(['。'])
        #print(s2_set)
        s2_batch, s2_len = get_batch(s2_set, word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        #print(s2_batch, s2_len)
        u = wenda_cod((s1_batch, s1_len))
        v = wenda_cod((s2_batch, s2_len))
        #print('v: ', v[0])
        features = torch.cat((u, v, torch.abs(u-v), u*v), 1).cuda()
        #print('cuda done！ ')
        t_features = wenda_infersent.classifier(features)[0]
        #print(t_features)
        #output = wenda_infersent.classifier(features)[0]
        #print(''.join(s1[idx]))
        #print(output)
        result_tensor.append(t_features)
    try:
        res = torch.cat(tuple(result_tensor)).reshape(len(s1),3)
        #print(res)
        #print(res.max(0))
        idx=res.max(0)[1][0]
        print('max idx: ',idx)
        print(''.join(s1[idx]))
        pd = time.time()
        print('elaps: ',pd-ps)
    except:
        pass
if __name__ == '__main__':        
    while(1):
        wenda_query()

