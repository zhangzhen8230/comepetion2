#coding=utf-8
import numpy as np
import pandas as pd
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
# import sys
# sys.path.append('.')

import sys
sys.path.append('../')
from feature_engineering.utils import DataUtil
from sklearn.cross_validation import train_test_split
from keras.callbacks import  EarlyStopping,ModelCheckpoint
from sklearn.metrics import log_loss
from sklearn.cross_validation import  StratifiedKFold
from keras.regularizers import l2
import keras.backend as K
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
MAX_LEN = 30
N_EPOCH = 10
N_FOLD = 5
sta1len = 54
sta2len = 14

def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    # f_pretrained_weights = open(pretrained_weights_path,'rb')
    # pretrained_weights = pickle.load(f_pretrained_weights)
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=trainable, **kwargs)
    return embedding

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
                             output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(pretrained_embedding='../data/index_embed_martix.npy',
                           projection_dim=300, projection_hidden=0, projection_dropout=0.,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=MAX_LEN):
    # Based on: https://arxiv.org/abs/1606.01933
    
    q1 = Input(name='q1',shape=(maxlen,))
    q2 = Input(name='q2',shape=(maxlen,))
    sta1 = Input(name='sta1',shape=(sta1len,))
    sta2 =Input(name='sta2',shape=(sta2len,))


    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding, 
                                            mask_zero=False)
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)
    
    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
                Dense(projection_hidden, activation=activation),
                Dropout(rate=projection_dropout),
            ])
    projection_layers.extend([
            Dense(projection_dim, activation=None),
            Dropout(rate=projection_dropout),
        ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)    
    
    # Compare
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)
    
    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
    # Classifier

    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    sta1dense = Dense(dense_dim,activation=activation)(sta1)
    sta2dense = Dense(dense_dim,activation=activation)(sta2)
    sta1dense = Dropout(dense_dropout)(sta1dense)
    sta1dense = BatchNormalization()(sta1dense)
    sta2dense = Dropout(dense_dropout)(sta2dense)
    sta2dense = BatchNormalization()(sta2dense)




    dense = Concatenate()([dense,sta1dense,sta2dense])
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[q1, q2,sta1,sta2], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', 
                  metrics=['binary_crossentropy','accuracy'])
    return model
def result_describe(path):
    '''
    :分析测试结果
    '''
    df_result = pd.read_table(path,names=['score'])
    print(df_result['score'].describe())
def train():
    "训练模型"

    Train_left = np.load('../data/X_train_question1.npy')
    Train_right = np.load('../data/X_train_question2.npy')
    Train_label= np.load('../data/train_label.npy')
    Train_label = Train_label.astype(np.int64)
    stack_tr = np.zeros((Train_label.shape[0]))
    statistics_feature = DataUtil.load_matrix('../feature_train/feature_min_max.txt')
    sta2 = pd.read_csv('../feature_train/feature_deepnet.csv').values
    for k,(tr,va) in enumerate(StratifiedKFold(Train_label,random_state=27,n_folds=N_FOLD)):
        model =  decomposable_attention()
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
        checkpoint = ModelCheckpoint('../model_file/attenion1_{}.hdf5'.format(k), monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
        early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        callbacks_list = [checkpoint, early]
        model.fit([X_train_left1,X_train_right1,train_stistics1,train_sta1], Y_train1, batch_size=128, epochs=N_EPOCH, verbose=1,
                 validation_data=([X_train_left2,X_train_right2,train_stistics2,train_sta2], Y_train2),callbacks=callbacks_list)
        model.load_weights('../model_file/attenion1_{}.hdf5'.format(k))
        val_pre = model.predict([X_val_left,X_val_right,val_stistics,val_sta2]).flatten()

        print(val_pre.shape,val_pre)
        stack_tr[va] += val_pre
        print('log_loss',log_loss(Y_val,val_pre))
    df_train_result = pd.DataFrame({'Score':stack_tr})
    df_train_result.to_csv('../result/attention1_train.txt',header=False,index=False)
def test():
    "预测得分"
    version = 'dec'
    model =  decomposable_attention()
    test_left = np.load('../data/test_left.npy')
    test_right = np.load('../data/test_right.npy')
    statistics_feature = DataUtil.load_matrix('../feature_test/feature_min_max.txt')
    sta2 = pd.read_csv('../feature_test/feature_deepnet.csv').values
    result_np = np.zeros((len(test_right,)))
    for i in range(N_FOLD):
        model.load_weights('../model_file/attenion1_{}.hdf5'.format(i))
        score = model.predict([test_left,test_right,statistics_feature,sta2])
        score = np.reshape(score,(len(score),))
        result_np +=score
    result_df = pd.DataFrame({"score":result_np/N_FOLD})
    import datetime
    unquie_flag = datetime.datetime.now().strftime('%m_%d_%H_%M')
    result_df.to_csv('../result/attention1_test.txt',index=False,header=False)

if __name__ == '__main__':
   train()
   test()
    # result_describe('./result/result_add_vec.txt')
#

