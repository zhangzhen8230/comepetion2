
# coding: utf-8

# In[12]:


import numpy as np
import os, json, re
from collections import Counter
import Levenshtein

from gensim.models import Word2Vec

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
import gensim
from keras.preprocessing.sequence import pad_sequences
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle as pk
def clean_str(stri):
    text = stri.lower()
    text = re.sub(r":", " : ", text)
    text = re.sub(r'\?', " ? ", text)
    text = re.sub(r'？', " ? ", text)
    text = re.sub(r'\xa1', " ¿ ", text)
    text = re.sub(r'\xbf', " ¿ ", text)
    text = re.sub(r'¿', " ¿ ", text)
    text = re.sub(r'¡', " ¡ ", text)
    text = re.sub(r'\n', " ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ' ", text)
    text = re.sub(r'"', " " , text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"2016", "año", text)
    text = re.sub(r"2017", "año", text)
    text = re.sub(r"\*+", " * ", text)
    text = re.sub(r"[`|´]", " ", text)
    text = re.sub(r"[&|#|}]", " ", text)
    text = re.sub(r"[&|#|}]", " ", text)
    return text

def log_loss(y_test, pred):
    loss = 0.
    for i, j in zip(pred, y_test):
        if j == 1:
            loss += np.log(i)[0]
        else:
            loss += np.log(1-i)[0]
    return (-loss / len(pred))


# In[7]:


fi1 = open('../data/cikm_english_train_20180516.txt').readlines()
fi2 = open('../data/cikm_spanish_train_20180516.txt').readlines()
fi3 = open('../data/cikm_test_a_20180516.txt').readlines()
fi4 = open('../data/cikm_unlabel_spanish_train_20180516.txt').readlines()

en1 = []
en2 = []
es1 = []
es2 = []

out = []

esu = []
enu = []

test1, test2 = [], []


# In[8]:


#load data
for s in fi1:
    #truncate 
    s1, s2, s3, s4, res = [clean_str(fuck) for fuck in s.split('\t')]
    en1 += [[k for k in s1.split(' ') if k != '']]
    es1 += [[k for k in s2.split(' ') if k != '']]
    en2 += [[k for k in s3.split(' ') if k != '']]
    es2 += [[k for k in s4.split(' ') if k != '']]
    out  += [int(res[:1])]
for s in fi2:
    s1, s2, s3, s4, res = [clean_str(fuck) for fuck in s.split('\t')]
    es1 += [[k for k in s1.split(' ') if k != '']]
    en1 += [[k for k in s2.split(' ') if k != '']]
    es2 += [[k for k in s3.split(' ') if k != '']]
    en2 += [[k for k in s4.split(' ') if k != '']]
    out  += [int(res[:1])]

for s in fi3:
    s1, s2 = [clean_str(fuck) for fuck in s.split('\t')]
    test1 += [[k for k in s1.split(' ') if k != '']]
    test2 += [[k for k in s2.split(' ') if k != '']]

for s in fi4:
    s1, s2 = [clean_str(fuck) for fuck in s.split('\t')]
    esu += [[k for k in s1.split(' ') if k != '']]
    enu += [[k for k in s2.split(' ') if k != '']]


# In[ ]:


#load spanish vocabulary
es_wiki_w2v = open('../data/wiki.es.vec')
line = es_wiki_w2v.readline()
es_voc = {}
while line:
    line = es_wiki_w2v.readline()
    idx = line.find(' ')
    word = line[:idx]
    es_voc[word] = True
#find all the words that not appear in dictionary.    
unf = {}
for es in [es1, es2, esu, test1, test2]:
    for s in es:
        for w in s:
            if w not in es_voc:
                if w not in unf:
                    unf[w] = 1
                else:
                    unf[w] += 1
#create spanish error_correction dict
spanish_map = {}
for k in sorted(unf.keys(), key=lambda k: -unf[k]):
    dis = 10000
    kw = ''
    for w in es_voc:
        d = float(Levenshtein.distance(k, w)) / (len(k) + len(w))
        if d < dis:
            dis = d
            kw = w
    spanish_map[k] = kw
    print(k)
    print(kw)
    print(dis)
    print('----------')
pk.dump(spanish_map, open('../data/spanish_map.pk', 'wb'))


# In[15]:


#load english vocabulary
en_wiki_w2v = open('../data/wiki.en.vec')
line = en_wiki_w2v.readline()
en_voc = {}
while line:
    line = en_wiki_w2v.readline()
    idx = line.find(' ')
    word = line[:idx]
    en_voc[word] = True 
#find all the words that not appear in dictionary.    
unf = {}
for en in [en1, en2, enu]:
    for s in en:
        for w in s:
            if w not in en_voc:
                if w not in unf:
                    unf[w] = 1
                else:
                    unf[w] += 1
#create spanish error_correction dict
english_map = {}
for k in sorted(unf.keys(), key=lambda k: -unf[k]):
    dis = 10000
    kw = ''
    for w in en_voc:
        d = float(Levenshtein.distance(k, w)) / (len(k) + len(w))
        if d < dis:
            dis = d
            kw = w
    english_map[k] = kw
    print(k)
    print(kw)
    print(dis)
    print('----------')
pk.dump(english_map, open('../data/english_map.pk', 'wb'))


# In[16]:


en1 = [[english_map[w] if w in english_map else w for w in s] for s in en1]
en2 = [[english_map[w] if w in english_map else w for w in s] for s in en2]
enu = [[english_map[w] if w in english_map else w for w in s] for s in enu]

es1 = [[spanish_map[w] if w in spanish_map else w for w in s] for s in es1]
es2 = [[spanish_map[w] if w in spanish_map else w for w in s] for s in es2]
esu = [[spanish_map[w] if w in spanish_map else w for w in s] for s in esu]

test1 = [[spanish_map[w] if w in spanish_map else w for w in s] for s in test1]
test2 = [[spanish_map[w] if w in spanish_map else w for w in s] for s in test2]


# In[18]:


#only consider the words in our dataset. id = 0 means padding
es2id = {}
cnt = 1
for es in [es1, es2, esu, test1, test2]:
    for s in es:
        for w in s:
            if w in spanish_map:
                w = spanish_map[w]
            if w not in es2id:
                es2id[w] = cnt
                cnt += 1
en2id = {}
cnt = 1
for es in [en1, en2, enu]:
    for s in es:
        for w in s:
            if w in english_map:
                w = english_map[w]
            if w not in en2id:
                en2id[w] = cnt
                cnt += 1  


# In[25]:


#load spanish word2vec
es_wiki_w2v = open('../data/wiki.es.vec')
#first line static info
line = es_wiki_w2v.readline()
es2vec = [[] for i in range(len(es2id) + 1)]
es2vec[0] = np.zeros(300)
while line:
    line = es_wiki_w2v.readline()
    idx = line.find(' ')
    word = line[:idx]
    if word in es2id:
        es2vec[es2id[word]] = [float(i) for i in line[idx+1 : -2].split(' ')]


# In[26]:


#load english word2vec
en_wiki_w2v = open('../data/wiki.en.vec')
line = en_wiki_w2v.readline()
en2vec = [[] for i in range(len(en2id) + 1)]
en2vec[0] = np.zeros(300)
while line:
    line = en_wiki_w2v.readline()
    idx = line.find(' ')
    word = line[:idx]
    if word in en2id:
        en2vec[en2id[word]] = [float(i) for i in line[idx+1 : -2].split(' ')]


# In[22]:


en1 = pad_sequences([[en2id[w] for w in s] for s in en1], maxlen=48, 
                    padding='post', truncating='pre')
en2 = pad_sequences([[en2id[w] for w in s] for s in en2], maxlen=48, 
                    padding='post', truncating='pre')
enu = pad_sequences([[en2id[w] for w in s] for s in enu], maxlen=48, 
                    padding='post', truncating='pre')

es1 = pad_sequences([[es2id[w] for w in s] for s in es1], maxlen=48, 
                    padding='post', truncating='pre')
es2 = pad_sequences([[es2id[w] for w in s] for s in es2], maxlen=48, 
                    padding='post', truncating='pre')
esu = pad_sequences([[es2id[w] for w in s] for s in esu], maxlen=48, 
                    padding='post', truncating='pre')

test1 = pad_sequences([[es2id[w] for w in s if w] for s in test1], 
                      maxlen=48, padding='post', truncating='pre')
test2 = pad_sequences([[es2id[w] for w in s if w] for s in test2], 
                      maxlen=48, padding='post', truncating='pre')


# In[27]:


pk.dump(en1, open('../data/en1.dat', 'wb'))
pk.dump(en2, open('../data/en2.dat', 'wb'))
pk.dump(enu, open('../data/enu.dat', 'wb'))

pk.dump(es1, open('../data/es1.dat', 'wb'))
pk.dump(es2, open('../data/es2.dat', 'wb'))
pk.dump(enu, open('../data/esu.dat', 'wb'))
pk.dump(out, open('../data/out.dat', 'wb'))

pk.dump(en2vec, open('../data/en2vec.dat', 'wb'))
pk.dump(es2vec, open('../data/es2vec.dat', 'wb'))
pk.dump([test1, test2], open('../data/test.dat', 'w'))


# In[28]:


pk.dump(es2id, open('../data/es2id.dat', 'w'))
pk.dump(en2id, open('../data/en2id.dat', 'w'))

