#coding=utf-8
import pandas as pd
import numpy as np
from keras.models import Model
import sys
sys.path.append('../')
from feature_engineering.utils import DataUtil
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,Concatenate
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
sta1_len = 54
sta2_len =14
MAXLEN = 30
def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    # f_pretrained_weights = open(pretrained_weights_path,'rb')
    # pretrained_weights = pickle.load(f_pretrained_weights)
    pretrained_weights = np.load(pretrained_weights_path)
    # in_dim, out_dim = pretrained_weights.shape
    # embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=trainable, **kwargs)
    return pretrained_weights

embedding_matrix = create_pretrained_embedding('../data/index_embed_martix.npy')
in_shape = embedding_matrix.shape[0]
filter_length = 5
nb_filter = 64
pool_length = 4

print('Build model...')
def deepnet_mode():
    q1 = Input(name='q1',shape=(MAXLEN,))
    q2 = Input(name='q2',shape=(MAXLEN,))
    sta1 = Input(name='sta1',shape=(sta1_len,))
    sta2 =Input(name='sta2',shape=(sta2_len,))

    embedding_q1 = Embedding(in_shape,
                     300,
                     weights=[embedding_matrix],
                     input_length=MAXLEN,
                     trainable=False)(q1)
    embedding_q2 = Embedding(in_shape,
                     300,
                     weights=[embedding_matrix],
                     input_length=MAXLEN,
                     trainable=False)(q2)
    model1 = TimeDistributed(Dense(300, activation='relu'))(embedding_q1)
    model1 = (Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))(model1)
    model2 = (TimeDistributed(Dense(300, activation='relu')))(embedding_q2)
    model2 = (Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))(model2)
    model3 = (Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))(embedding_q1)
    model3 = Dropout(0.2)(model3)
    model3 = Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(model3)

    model3 = GlobalMaxPooling1D()(model3)
    model3 = (Dropout(0.2))(model3)
    model3 = Dense(300)(model3)
    model3 = Dropout(0.2)(model3)
    model3 = BatchNormalization()(model3)
    model4 = Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(embedding_q2)
    model4 = (Dropout(0.2))(model4)
    model4 = (Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))(model4)

    model4 = GlobalMaxPooling1D()(model4)
    model4 = Dropout(0.2)(model4)

    model4 = Dense(300)(model4)
    model4 = Dropout(0.2)(model4)
    model4 = BatchNormalization()(model4)

    model5 = LSTM(300, dropout_W=0.2, dropout_U=0.2)(embedding_q1)
    model6 = LSTM(300, dropout_W=0.2, dropout_U=0.2)(embedding_q2)
    model7 = Dense(50,input_shape=(54,))(sta1)
    model8 = Dense(20,input_shape=(14,))(sta2)
    merged_model = Concatenate()([model1, model2, model3, model4, model5, model6,model7,model8])
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(300)(merged_model)
    merged_model = PReLU()(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(300)(merged_model)
    merged_model = PReLU()(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(300)(merged_model)
    merged_model = PReLU()(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(300)(merged_model)
    merged_model = PReLU()(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    merged_model = BatchNormalization()(merged_model)

    merged_model = Dense(1)(merged_model)
    merged_model = Activation('sigmoid')(merged_model)
    model = Model(inputs=[q1, q2,sta1,sta2], outputs=merged_model)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


checkpoint = ModelCheckpoint('weights.h5', monitor='logloss', save_best_only=True, verbose=2)
merged_model = deepnet_mode()
merged_model.save_weights('../model_file/deepnet_weight_init.hdf5')
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
    for k,(tr,va) in enumerate(StratifiedKFold(Train_label,random_state=27,n_folds=5)):
        merged_model.load_weights('../model_file/deepnet_weight_init.hdf5')
        print(' stack:{}/{}'.format(k+1,5))
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
        checkpoint = ModelCheckpoint('../model_file/deepnet1_{}.hdf5'.format(k), monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
        early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        callbacks_list = [checkpoint, early]
        merged_model.fit([X_train_left1,X_train_right1,train_stistics1,train_sta1], Y_train1, batch_size=128, epochs=20, verbose=1,
                 validation_data=([X_train_left2,X_train_right2,train_stistics2,train_sta2], Y_train2),callbacks=callbacks_list)
        merged_model.load_weights('../model_file/deepnet1_{}.hdf5'.format(k))
        val_pre = merged_model.predict([X_val_left,X_val_right,val_stistics,val_sta2]).flatten()

        print(val_pre.shape,val_pre)
        stack_tr[va] += val_pre
        print('log_loss',log_loss(Y_val,val_pre))
    df_train_result = pd.DataFrame({'Score':stack_tr})
    df_train_result.to_csv('../result/deepnet1_train.txt',header=False,index=False)
def test():
    "预测得分"
    test_left = np.load('../data/test_left.npy')
    test_right = np.load('../data/test_right.npy')
    statistics_feature = DataUtil.load_matrix('../feature_test/feature_min_max.txt')
    sta2 = pd.read_csv('../feature_test/feature_deepnet.csv').values
    result_np = np.zeros((len(test_right,)))
    for i in range(5):
        merged_model.load_weights('../model_file/deepnet1_{}.hdf5'.format(i))
        score = merged_model.predict([test_left,test_right,
                               statistics_feature,sta2])
        score = np.reshape(score,(len(score),))
        result_np +=score
    result_df = pd.DataFrame({"score":result_np/5})
    import datetime
    result_df.to_csv('../result/deepnet1_test.txt',index=False,header=False)

if __name__ == '__main__':
   train()
   test()
    # result_describe('./result/result_add_vec.txt')
#



