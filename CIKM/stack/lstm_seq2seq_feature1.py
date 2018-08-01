# encoding: utf-8

"""
@author: mario
@date: 2018/5/30
模型训练方法(所有的大写转换为小写字母)
"""
from keras.layers.advanced_activations import PReLU
from time import time
import pandas as pd
import numpy as np
# from gensim.corpora.dictionary import Dictionary
# from gensim.models import KeyedVectors
# import codecs
# from sklearn.model_selection import train_test_split

# import yaml

# from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.layers import Concatenate,Dropout,BatchNormalization
from keras.models import Model
from keras.layers import Input,Dense, Embedding, LSTM, Activation
import keras.backend as K
import  sys
sys.path.append('../')
from feature_engineering.utils import DataUtil
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
sta1_len = 54
sta2_len =14
embedim = 300
maxlen = 30
n_hidden = 300
batchSize = 128
n_epoch = 20
gradient_clipping_norm = 1.25

def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    # f_pretrained_weights = open(pretrained_weights_path,'rb')
    # pretrained_weights = pickle.load(f_pretrained_weights)
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=trainable, **kwargs)
    return embedding
def exponent_neg_manhattan_distance(left, right):
    """Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

def siamse_lstm():
    """定义网络模型结构"""
    # The visible layer
    left_input = Input(shape=(maxlen,), dtype='int32')
    right_input = Input(shape=(maxlen,), dtype='int32')
    sta1 = Input(name='sta1',shape=(sta1_len,))
    sta2 =Input(name='sta2',shape=(sta2_len,))
    EmbeddingLayer = create_pretrained_embedding('../data/index_embed_martix.npy')
    # embedding_layer = Embedding(input_dim=index_embed_matrix.shape[0], output_dim=embedim, weights=[index_embed_matrix], trainable=True)
    # Embedded version of the inputs
    encoded_left = EmbeddingLayer(left_input)
    encoded_right = EmbeddingLayer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = LSTM(n_hidden, dropout=0.1, return_sequences=False)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)
    sta1dense = Dense(n_hidden,activation='elu')(sta1)
    sta2dense = Dense(n_hidden,activation='elu')(sta2)
    sta1dense = Dropout(0.2)(sta1dense)
    sta1dense = BatchNormalization()(sta1dense)
    sta2dense = Dropout(0.2)(sta2dense)
    sta2dense = BatchNormalization()(sta2dense)
    # Pack it all up into a model
    merged_model = Concatenate()([left_output, right_output, sta1dense, sta2dense])
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(300)(merged_model)
    merged_model = PReLU()(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(1)(merged_model)
    merged_model = Activation('sigmoid')(merged_model)
    model = Model(inputs=[left_input, right_input,sta1,sta2], outputs=merged_model)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
checkpoint = ModelCheckpoint('weights.h5', monitor='logloss', save_best_only=True, verbose=2)
merged_model = siamse_lstm()
merged_model.save_weights('../model_file/siamse_lstm_init.hdf5')
def train():
    "训练模型"
    Train_left = np.load('../data/X_train_question1.npy')
    Train_right = np.load('../data/X_train_question2.npy')
    Train_label= np.load('../data/train_label.npy')
    Train_label = Train_label.astype(np.int64)
    stack_tr = np.zeros((Train_label.shape[0]))
    statistics_feature = DataUtil.load_matrix('../feature_train/feature_min_max.txt')
    sta2 = pd.read_csv('../feature_train/feature_deepnet.csv').values
    from sklearn.cross_validation import StratifiedKFold,train_test_split
    from sklearn.metrics import log_loss
    N_FOLD = 5
    for k,(tr,va) in enumerate(StratifiedKFold(Train_label,random_state=27,n_folds=N_FOLD)):
        merged_model.load_weights('../model_file/siamse_lstm_init.hdf5')
        print(' stack:{}/{}'.format(k+1,N_FOLD))
        X_train_left = Train_left[tr]
        X_train_right = Train_right[tr]
        Y_train = Train_label[tr]
        train_stistics = statistics_feature[tr]
        train_sta = sta2[tr]
        val_sta2 = sta2[va]
        val_stistics = statistics_feature[va]
        X_val_left = Train_left[va]
        X_val_right = Train_right[va]
        Y_val = Train_label[va]
        X_train_left1,X_train_left2,X_train_right1,X_train_right2,train_stistics1,train_stistics2,train_sta1,train_sta2,Y_train1,Y_train2 = \
            train_test_split(X_train_left,X_train_right,train_stistics,train_sta,Y_train,test_size=0.2,stratify=Y_train)
        print ("Train...")
        checkpoint = ModelCheckpoint('../model_file/lstm1_{}.hdf5'.format(k), monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
        early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        callbacks_list = [checkpoint, early]
        merged_model.fit([X_train_left1,X_train_right1,train_stistics1,train_sta1], Y_train1, batch_size=128, epochs=n_epoch, verbose=1,
                 validation_data=([X_train_left2,X_train_right2,train_stistics2,train_sta2], Y_train2),callbacks=callbacks_list)
        merged_model.load_weights('../model_file/lstm1_{}.hdf5'.format(k))
        val_pre = merged_model.predict([X_val_left,X_val_right,val_stistics,val_sta2]).flatten()
        print(val_pre.shape,val_pre)
        stack_tr[va] += val_pre
        print('log_loss',log_loss(Y_val,val_pre))
    df_train_result = pd.DataFrame({'Score':stack_tr})
    df_train_result.to_csv('../result/lstm1_train.txt',header=False,index=False)
def test():
    "预测得分"
    version = 'siamse_lstm'
    test_left = np.load('../data/test_left.npy')
    test_right = np.load('../data/test_right.npy')
    statistics_feature = DataUtil.load_matrix('../feature_test/feature_min_max.txt')
    sta2 = pd.read_csv('../feature_test/feature_deepnet.csv').values
    result_np = np.zeros((len(test_right,)))
    for i in range(5):
        merged_model.load_weights('../model_file/lstm1_{}.hdf5'.format(i))
        score = merged_model.predict([test_left,test_right,
                               statistics_feature,sta2])
        score = np.reshape(score,(len(score),))
        result_np +=score
    result_df = pd.DataFrame({"score":result_np/5})
    import datetime
    result_df.to_csv('../result/lstm1_test.txt',index=False,header=False)

if __name__ == '__main__':
   train()
   test()