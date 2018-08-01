import pandas as pd
import numpy as np
import re
import codecs
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
def generate():
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
    pos_train.to_csv('../data/pos.csv',index=False,encoding='utf-8')

def generate_data():
    df_pos = pd.read_csv('../data/pos.csv',encoding='utf-8')
    q2 = []
    question_set =[]
    df_pos = df_pos.sort(['question1'])
    df_pos.to_csv('../data/pos_df.csv',encoding='utf-8',index=False)
    q1 = df_pos.iloc[0,0]
    same = False
    for i in range(len(df_pos)):
        if(df_pos.iloc[i,0]==q1):
            q2.append(df_pos.iloc[i,1])
            same = True
        else:
            if same:
                question_set.append(q2)
                same = False
            q2 = [df_pos.iloc[i,1]]
            q1 = df_pos.iloc[i,0]
    q_new = []
    for q in question_set:
       q2 = [[q[i],q[i+1]] for i in range(len(q)) if i+1 < len(q)]
       for s in q2:
        q_new.append(s)
    df_generate_pos = pd.DataFrame(np.array(q_new),columns=['question1','question2'])
    df_generate_pos['label'] = 1
    df_generate_pos.to_csv('../data/generate_pos.csv',index=False,encoding='utf-8')
generate()
generate_data()