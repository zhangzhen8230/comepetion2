# encoding: utf-8

import re
import pickle
import os
import pandas as pd
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models import KeyedVectors
import codecs
from nltk.corpus import stopwords
from keras.preprocessing import sequence
# from nltk.stem import LancasterStemmer
import nltk
maxlen = 30
embedim = 300
def combine_data():
    """训练数据进行融合"""
    enPath = codecs.open('../data/cikm_english_train_20180516.txt', 'r', encoding='utf-8')
    esPath = codecs.open('../data/cikm_spanish_train_20180516.txt', 'r', encoding='utf-8')
    comPath = codecs.open('../data/cikm_ns_train.txt', 'w+', encoding='utf-8')
    with enPath as epath:
        for line in epath:
            segLine = line.strip().split('	')
            comPath.write(segLine[1] + '	' + segLine[3]+'	'+segLine[4]+'\n')
    with esPath as spath:
        for line in spath:
            segLine = line.strip().split('	')
            comPath.write(segLine[0] + '	' + segLine[2] + '	' + segLine[4] + '\n')

def read_data():
    """读取训练数据"""
    dtrain = codecs.open('../data/cikm_ns_train.txt', 'r', encoding='utf-8')
    sen1, sen2, label = [], [], []
    with dtrain as dt:
        for line in dt:
            segLine = line.strip().split('	')
            sen1.append(segLine[0])
            sen2.append(segLine[1])
            label.extend(segLine[2])
    df_sen1 = pd.DataFrame([word for word in sen1])
    df_sen2 = pd.DataFrame([word for word in sen2])
    df_label = pd.DataFrame([word for word in label])
    df_train1 = pd.DataFrame(pd.concat([df_sen1, df_sen2, df_label], axis=1))

    df_train1.columns = ['question1'] + ['question2'] + ['label']
    df_train1['question1'] = df_train1['question1'].map(lambda x: clean_str_stem(x))
    df_train1['question2'] = df_train1['question2'].map(lambda x: clean_str_stem(x))
    df_train1 = df_train1.drop_duplicates(subset=["question1","question2","label"],keep="first")
    pos_train = df_train1[df_train1['label']=='1']
    neg_train =  df_train1[df_train1['label']=='0']
    scale = ((len(pos_train) / (len(pos_train) + len(pos_train))) / 0.36) -1
    while scale > 1:
        pos_train = pd.concat([pos_train, pos_train])
        scale -=1
    pos_generate = pd.read_csv('../data/generate_pos.csv',encoding='utf-8')
    # pos_train = pd.concat([pos_train, pos_train.sample(int(len(pos_train)*scale),replace=True)])
    pos_train = pd.concat([pos_train,pos_generate.sample(2500)])
    print(len(pos_train)/(len(pos_train)+len(neg_train)))
    df_train = pd.concat((pos_train,neg_train))
    df_train = df_train.sample(frac=1)
    df_train.to_csv('../data/train.csv',index=False,encoding='utf-8')
    return df_train

def clean_str_stem(stri):
    """将所有的大写字母转换为小写字母"""
    text = stri.lower()
    text = re.sub(r"[0-9]+", " ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r'\?', " ? ", text)
    text = re.sub(r'？', " ? ", text)
    text = re.sub(r'¿', " ¿ ", text)
    text = re.sub(r'¡', " ¡ ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r'"', " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"\*+", "*", text)
    text = re.sub(r"[`|´]", " ", text)
    text = re.sub(r"[&|#|}]", " ", text)
    return text

def create_dictoinaries(model=None):
    """创建词词向量字典(训练)"""
    if model is not None:
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(), allow_update= True)
        w2index = {v: k + 1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2index.keys()}
        return w2index, w2vec

def wor2vec_obtain(data):
    """获取西班牙语词向量"""
    print('start load vec')
    model = KeyedVectors.load_word2vec_format('../data/wiki.es.vec', binary=False)
    print('vec_load done')
    w2index, w2vec = create_dictoinaries(model=model)
    stops = set(stopwords.words("spanish"))
    for word in data:
        segList = word.strip().split(' ')
        for each in segList:
            if (each not in stops) and (each not in w2index.keys()):
                # print(len(w2index))
                # print(w2index)
                # print("不存在",each)
                w2index[each] = len(w2index)
                w2vec[each] = np.random.random((300,))
    return w2index, w2vec

def text2index(w2index, data, maxlen):
    """词转换为词索引，比如[中国、安徽、合肥]-->【1，5，30]"""
    stops = set(stopwords.words("spanish"))
    not_exit_word = []
    new_data = []
    for word in data:
        new_word = []
        segList = word.strip().split(' ')
        for each in segList:
            if (each not in stops) and (each != ''):
                try:
                    new_word.append(w2index[each])
                except:
                    not_exit_word.append(each)
                    new_word.append(0)
        new_data.append(new_word)
    print("不存在单词",len(not_exit_word),not_exit_word)
    new_data = sequence.pad_sequences(new_data, padding='post', maxlen=maxlen)
    return new_data

def train_data(w2index, w2vec, embed_dim):
    """获取所有词语的个数和词索引与词向量对应的矩阵"""
    word_count = len(w2index) + 1
    index_embed_matrix = np.zeros((word_count, embed_dim))
    for word, index in w2index.items():
        index_embed_matrix[index, :] = w2vec[word]
    return word_count, index_embed_matrix

def train():
    """训练数据"""
    df_train = read_data()
    question_col = ['question1', 'question2']
    X_train = df_train[question_col]
    Y_train = df_train['label']
    df_test = pd.read_table('../data/cikm_test_b_20180730.txt',names=['question1','question2'])
    df_test['question1'] = df_test['question1'].map(lambda x: clean_str_stem(x))
    df_test['question2'] = df_test['question2'].map(lambda x:clean_str_stem(x))
    if(os.path.exists('../data/w2index.txt')):
        print('w2index has exited,starting loading')
        f_w2index = open('../data/w2index.txt','rb')
        f_w2vec = open('../data/w2index.txt','rb')
        w2index = pickle.load(f_w2index)
        w2vec = pickle.load(f_w2vec)
    else:
        print('w2index has not exited,starting loading')
        data = np.concatenate((X_train.question1,X_train.question2,df_test.question1,df_test.question2),axis=0)
        w2index, w2vec = wor2vec_obtain(data)
        with open ('../data/w2index.txt','wb') as f_w2index:
            pickle.dump(w2index,f_w2index)
        with open ('../data/w2vec.txt','wb') as f_w2vec:
            pickle.dump(w2vec,f_w2vec)
    X_train_question1 = text2index(w2index,X_train.question1, maxlen)
    X_train_question2 = text2index(w2index, X_train.question2, maxlen)
    np.save('../data/X_train_question1.npy',X_train_question1)
    np.save('../data/X_train_question2.npy',X_train_question2)
    np.save('../data/train_label.npy',Y_train)
    if(os.path.exists('../data/index_embed_martix.txt')):
        index_embed_martix =  np.load('../data/index_embed_martix.npy')
    else:
        word_count, index_embed_martix = train_data(w2index, w2vec, embedim)
        print('start loading numpy martix')
        np.save('../data/index_embed_martix.npy',index_embed_martix)
    print(index_embed_martix.shape)
def test():
    if(os.path.exists('../data/w2index.txt')):
        f_w2index = open('../data/w2index.txt','rb')
        w2index = pickle.load(f_w2index)
    else:
        w2index, w2vec = wor2vec_obtain()
        with open ('../data/w2index.txt','wb') as f_w2index:
            pickle.dump(w2index,f_w2index)
        with open ('../data/w2vec.txt','wb') as f_w2vec:
            pickle.dump(w2vec,f_w2vec)
    df_test = pd.read_table('../data/cikm_test_b_20180730.txt',names=['question1','question2'])
    df_test.to_csv('../data/test.csv',index=False,encoding='utf-8')
    df_test['question1'] = df_test['question1'].map(lambda x: clean_str_stem(x))
    df_test['question2'] = df_test['question2'].map(lambda x:clean_str_stem(x))
    x_test_question1 = text2index(w2index,df_test.question1,maxlen)
    x_test_question2 = text2index(w2index,df_test.question2,maxlen)
    np.save('../data/test_left.npy',x_test_question1)
    np.save('../data/test_right.npy',x_test_question2)
if __name__ == '__main__':
    # generate()
    train()
    test()
