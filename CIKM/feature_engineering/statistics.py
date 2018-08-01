# -*- coding: utf-8 -*-

import math
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_engineering.utils import NgramUtil,DistanceUtil,DataUtil,MathUtil
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import codecs
MISSING_VALUE_NUMERIC = -1
def extract_no( row):
    q1 = str(row['question1']).strip().lower()
    q2 = str(row['question2']).strip().lower()

    not_cnt1 = q1.count('no')
    not_cnt2 = q2.count('no')
    fs = list()
    fs.append(not_cnt1)
    fs.append(not_cnt2)
    if not_cnt1 > 0 and not_cnt2 > 0:
        fs.append(1.)
    else:
        fs.append(0.)
    if (not_cnt1 > 0) or (not_cnt2 > 0):
        fs.append(1.)
    else:
        fs.append(0.)
    if not_cnt2 <= 0 < not_cnt1 or not_cnt1 <= 0 < not_cnt2:
        fs.append(1.)
    else:
        fs.append(0.)

    return fs
def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = q1words.get(word, 0) + 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = q2words.get(word, 0) + 1
    n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
    n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
    n_tol = sum(q1words.values()) + sum(q2words.values())
    if 1e-6 > n_tol:
        return [0.]
    else:
        return 1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol
def init_idf(data):
    idf = {}
    q_set = set()
    for index, row in data.iterrows():
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        if q1 not in q_set:
            q_set.add(q1)
            words = q1.lower().split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
        if q2 not in q_set:
            q_set.add(q2)
            words = q2.lower().split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
    num_docs = len(data)
    for word in idf:
        idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
    return idf

def tf_idf_word_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        q1words[word] = q1words.get(word, 0) + 1
    for word in str(row['question2']).lower().split():
        q2words[word] = q2words.get(word, 0) + 1
    sum_shared_word_in_q1 = sum([q1words[w] * idf.get(w, 0) for w in q1words if w in q2words])
    sum_shared_word_in_q2 = sum([q2words[w] * idf.get(w, 0) for w in q2words if w in q1words])
    sum_tol = sum(q1words[w] * idf.get(w, 0) for w in q1words) + sum(
        q2words[w] * idf.get(w, 0) for w in q2words)
    if 1e-6 > sum_tol:
        return [0.]
    else:
        return 1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol
def len_word_sentence_feature(row):
    q1 = str(row['question1'])
    q2 = str(row['question2'])
    fs = list()
    # fs.append(len(q1))
    # fs.append(len(q2))
    fs.append(len(q1.split()))
    fs.append(len(q2.split()))
    return fs

def lengthdiff( row):
    q1 = row['question1']
    q2 = row['question2']
    return [abs(len(q1.split()) - len(q2.split()))]


def LengthDiffRate( row):
    len_q1 = len(row['question1'].split())
    len_q2 = len(row['question2'].split())
    if max(len_q1, len_q2) < 1e-6:
        return [0.0]
    else:
        return [1.0 * min(len_q1, len_q2) / max(len_q1, len_q2)]
def init_tfidf(df_all):
    tfidf = TfidfVectorizer(stop_words=stops, ngram_range=(1, 1))
    tfidf_txt = pd.Series(
        df_all['question1'].tolist() + df_all['question2'].tolist()).astype(str)
    tfidf.fit_transform(tfidf_txt)
    return tfidf

def extract_tfidf_feature(row):
    q1 = str(row['question1'])
    q2 = str(row['question2'])
    fs = list()
    fs.append(np.sum(tfidf.transform([str(q1)]).data))
    fs.append(np.sum(tfidf.transform([str(q2)]).data))
    fs.append(np.mean(tfidf.transform([str(q1)]).data))
    fs.append(np.mean(tfidf.transform([str(q2)]).data))
    fs.append(len(tfidf.transform([str(q1)]).data))
    fs.append(len(tfidf.transform([str(q2)]).data))
    return fs

def generate_dul_num(data_all):
    dul_num = {}
    for index, row in data_all.iterrows():
        q1 = str(row.question1).strip()
        q2 = str(row.question2).strip()
        dul_num[q1] = dul_num.get(q1, 0) + 1
        if q1 != q2:
            dul_num[q2] = dul_num.get(q2, 0) + 1
    return dul_num

def extract_dul_num( row):
    q1 = str(row['question1']).strip()
    q2 = str(row['question2']).strip()

    dn1 = dul_num[q1]
    dn2 = dul_num[q2]
    return [dn1, dn2, max(dn1, dn2), min(dn1, dn2)]

def extract_NgramDiceDistance(row):
    # print(str(row['question1']))
    q1 = str(row['question1']).strip().lower().split()
    q2 = str(row['question2']).strip().lower().split()
    fs = list()
    for n in range(1, 4):
        q1_ngrams = NgramUtil.ngrams(q1, n)
        q2_ngrams = NgramUtil.ngrams(q2, n)
        fs.append(DistanceUtil.dice_dist(q1_ngrams, q2_ngrams))
    return fs
def extract_NgramJaccardCoef(row):
    # print(str(row['question1']))
    q1 = str(row['question1']).strip().lower().split()
    q2 = str(row['question2']).strip().lower().split()
    fs = list()
    for n in range(1, 4):
        q1_ngrams = NgramUtil.ngrams(q1, n)
        q2_ngrams = NgramUtil.ngrams(q2, n)
        fs.append(DistanceUtil.jaccard_coef(q1_ngrams, q2_ngrams))
    return fs


#
def extract_edit_Distance(row):
    q1_stem =  str(row['question1']).strip().lower().split()
    q2_stem =  str(row['question2']).strip().lower().split()
    return [DistanceUtil.edit_dist(q1_stem, q2_stem)]
def extract_compression_Distance(row):
    q1_stem =  str(row['question1']).strip().lower()
    q2_stem =  str(row['question2']).strip().lower()
    return [DistanceUtil.compression_dist(q1_stem, q2_stem)]
def extract_ngramDistance(row):
    print(row)
    q1_words =  str(row['question1']).strip().lower().split()
    q2_words =  str(row['question2']).strip().lower().split()
    fs = list()
    for n_ngram in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n_ngram)
            q2_ngrams = NgramUtil.ngrams(q2_words, n_ngram)
            val_list = list()
            aggregation_modes_outer = ["mean", "max", "min", "median"]
            aggregation_modes_inner = ["mean", "std", "max", "min", "median"]
            for w1 in q1_ngrams:
                _val_list = list()
                for w2 in q2_ngrams:
                    s = DistanceUtil.edit_dist(w1, w2)
                    _val_list.append(s)
                if len(_val_list) == 0:
                    _val_list = [MISSING_VALUE_NUMERIC]
                val_list.append(_val_list)
            if len(val_list) == 0:
                val_list = [[MISSING_VALUE_NUMERIC]]

            for mode_inner in aggregation_modes_inner:
                tmp = list()
                for l in val_list:
                    tmp.append(MathUtil.aggregate(l, mode_inner))
                fs.extend(MathUtil.aggregate(tmp, aggregation_modes_outer))
            return fs
#
#     def get_feature_num(self):
#         return 2


def generate_powerful_word(train_subset_data):
    """
    计算数据中词语的影响力，格式如下：
        词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
    """
    words_power = {}
    for index, row in train_subset_data.iterrows():
        label = int(row['label'])
        q1_words = str(row['question1']).lower().split()
        q2_words = str(row['question2']).lower().split()
        all_words = set(q1_words + q2_words)
        q1_words = set(q1_words)
        q2_words = set(q2_words)
        for word in all_words:
            if word not in words_power:
                words_power[word] = [0. for i in range(7)]
            # 计算出现语句对数量
            words_power[word][0] += 1.
            words_power[word][1] += 1.

            if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                # 计算单侧语句数量
                words_power[word][3] += 1.
                if 1 == label:
                    # 计算正确语句对数量
                    words_power[word][2] += 1.
                    # 计算单侧语句正确比例
                    words_power[word][4] += 1.
            if (word in q1_words) and (word in q2_words):
                # 计算双侧语句数量
                words_power[word][5] += 1.
                if 1 == label:
                    # 计算正确语句对数量
                    words_power[word][2] += 1.
                    # 计算双侧语句正确比例
                    words_power[word][6] += 1.
    for word in words_power:
        # 计算出现语句对比例
        words_power[word][1] /= len(train_subset_data)
        # 计算正确语句对比例
        words_power[word][2] /= words_power[word][0]
        # 计算单侧语句对正确比例
        if words_power[word][3] > 1e-6:
            words_power[word][4] /= words_power[word][3]
        # 计算单侧语句对比例
        words_power[word][3] /= words_power[word][0]
        # 计算双侧语句对正确比例
        if words_power[word][5] > 1e-6:
            words_power[word][6] /= words_power[word][5]
        # 计算双侧语句对比例
        words_power[word][5] /= words_power[word][0]
    sorted_words_power = sorted(words_power.items(), key=lambda d: d[1][0], reverse=True)
    return sorted_words_power
def save_powerful_word(words_power, fp):
    f = open(fp, 'w',encoding='utf-8')
    for ele in words_power:
        f.write("%s" % ele[0])
        for num in ele[1]:
            f.write("\t%.5f" % num)
        f.write("\n")
    f.close()
def load_powerful_word(fp):
    powful_word = []
    f = open(fp, 'r',encoding='utf-8')
    for line in f:
        subs = line.split('\t')
        word = subs[0]
        stats = [float(num) for num in subs[1:]]
        powful_word.append((word, stats))
    f.close()
    return powful_word
def init_powerful_word_dside(pword, thresh_num, thresh_rate):
    pword_dside = []
    pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
    pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
    pword_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
    return pword_dside
def extract_powerful_word_dside(row):
    tags = []
    q1_words = str(row['question1']).lower().split()
    q2_words = str(row['question2']).lower().split()
    for word in pword_dside:
        if (word in q1_words) and (word in q2_words):
            tags.append(1.0)
        else:
            tags.append(0.0)
    return tags

def extract_PowerfulWordDoubleSideRate(row):
        num_least = 50
        rate = [1.0]
        q1_words = set(str(row['question1']).lower().split())
        q2_words = set(str(row['question2']).lower().split())
        share_words = list(q1_words.intersection(q2_words))
        for word in share_words:
            if word not in pword_dict:
                continue
            if pword_dict[word][0] * pword_dict[word][5] < num_least:
                continue
            rate[0] *= (1.0 - pword_dict[word][6])

        rate = [1 - num for num in rate]
        return rate
def extract_PowerfulWordOneSideRate(row):
    num_least = 50
    rate = [1.0]
    q1_words = set(str(row['question1']).lower().split())
    q2_words = set(str(row['question2']).lower().split())
    q1_diff = list(set(q1_words).difference(set(q2_words)))
    q2_diff = list(set(q2_words).difference(set(q1_words)))
    all_diff = set(q1_diff + q2_diff)
    for word in all_diff:
        if word not in pword_dict:
            continue
        if pword_dict[word][0] * pword_dict[word][3] < num_least:
            continue
        rate[0] *= (1.0 - pword_dict[word][4])
    rate = [1 - num for num in rate]
    return rate

def save_feature(df,step):
    # powerful_word_oside_feature = df.apply(extract_powerful_word_oside,axis=1)
    # DataUtil.save_matrix('../feature_{}/powerful_word_oside_feature.txt'.format(step),powerful_word_oside_feature,'w')
    PowerfulWordDoubleSideRate_feature = df.apply(extract_PowerfulWordDoubleSideRate,axis=1)
    DataUtil.save_matrix('../feature_{}/PowerfulWordDoubleSideRate_feature.txt'.format(step),PowerfulWordDoubleSideRate_feature,'w')
    # PowerfulWordOneSideRate_feautre = df.apply(extract_PowerfulWordOneSideRate,axis=1)
    # DataUtil.save_matrix('../feature_{}/PowerfulWordOneSideRate_feautre.txt'.format(step),PowerfulWordOneSideRate_feautre,'w')

    powerful_word_dside_feature = df.apply(extract_powerful_word_dside,axis=1).values
    DataUtil.save_matrix('../feature_{}/powerful_word_dside_feature.txt'.format(step),powerful_word_dside_feature,'w')
    ngramDistance_feature = df.apply(extract_ngramDistance,axis=1).values
    DataUtil.save_matrix('../feature_{}/ngramDistance_feature_feature.txt'.format(step),ngramDistance_feature,'w')
    # # print(ngramDistance_feature)
    NgramDiceDistance_feature = df.apply(extract_NgramDiceDistance,axis=1).values
    DataUtil.save_matrix('../feature_{}/NgramDiceDistance_feature.txt'.format(step),NgramDiceDistance_feature,'w')
    NgramJaccardCoef_feature = df.apply(extract_NgramJaccardCoef,axis=1).values
    DataUtil.save_matrix('../feature_{}/NgramJaccardCoef_feature.txt'.format(step),NgramJaccardCoef_feature,'w')
    Distance_feature = df.apply(extract_edit_Distance,axis=1)
    DataUtil.save_matrix('../feature_{}/Distance_feature.txt'.format(step),Distance_feature,'w')
    no_feature = df.apply(extract_no,axis=1)
    DataUtil.save_matrix('../feature_{}/feature_no.txt'.format(step),no_feature,'w')
    # '''word-match-feature'''
    word_match = df.apply(word_match_share,axis=1)
    DataUtil.save_vector('../feature_{}/word_match.txt'.format(step),word_match,'w')
    # '''tf-idf-word-share'''
    tf_idf_word_share_feature = df.apply(tf_idf_word_share,axis=1)
    DataUtil.save_vector('../feature_{}/tf_idf_word_share_feature.txt'.format(step),tf_idf_word_share_feature,'w')
    # '''len-feature'''
    print("start")
    len_feature = df.apply(len_word_sentence_feature,axis=1).values
    lendiff_feature = df.apply(lengthdiff,axis=1)
    lendiffrate_feature = df.apply(LengthDiffRate,axis=1)
    DataUtil.save_matrix('../feature_{}/len_feature.txt'.format(step),len_feature,'w')
    DataUtil.save_matrix('../feature_{}/lendiff_feature.txt'.format(step),lendiff_feature,'w')
    DataUtil.save_matrix('../feature_{}/lendiffrate_feature.txt'.format(step),lendiffrate_feature,'w')

    tfidf_feature= df.apply(extract_tfidf_feature,axis=1)
    DataUtil.save_matrix('../feature_{}/tfidf_feature.txt'.format(step),tfidf_feature,'w')
    # '''dul_num_feature'''
    # print('start load')
    dul_num = df.apply(extract_dul_num,axis=1)
    DataUtil.save_matrix('../feature_{}/dul_num.txt'.format(step),dul_num,'w')

'''
初始化数据，保存特征
'''

stops = set(stopwords.words("spanish"))

df_train = pd.read_csv('../data/train.csv',encoding='utf-8')
df_test = pd.read_csv('../data/test.csv',encoding='utf-8')
df_all = pd.concat((df_train,df_test))
pword = generate_powerful_word(df_train)
# pword = load_powerful_word('../data/powerful_word.txt')
pword_dside = init_powerful_word_dside(pword,100,0.85)
# pword_oside = init_powerful_word_oside(pword,500,0.8)
pword_dict = dict(pword)
idf = init_idf(df_all)
tfidf = init_tfidf(df_all)
dul_num = generate_dul_num(df_all)
save_feature(df_train,'train')
save_feature(df_test,'test')