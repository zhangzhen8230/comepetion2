from feature_engineering.utils import DataUtil
import numpy as np
from  sklearn import preprocessing
import pandas as pd
# from keras.models import pa

import codecs
from sklearn.cross_validation import train_test_split
# df = read_data()
# label = df.label.values
def save_feature(step):
    '''path'''
    NgramDiceDistance_feature_path ='../feature_{}/NgramDiceDistance_feature.txt'.format(step)
    NgramJaccardCoef_feature_path = '../feature_{}/NgramJaccardCoef_feature.txt'.format(step)
    Distance_feature_path = '../feature_{}/Distance_feature.txt'.format(step)
    no_feature_path = '../feature_{}/feature_no.txt'.format(step)
    word_match_path = '../feature_{}/word_match.txt'.format(step)
    tf_idf_word_share_feature_path = '../feature_{}/tf_idf_word_share_feature.txt'.format(step)
    len_feature_path = '../feature_{}/len_feature.txt'.format(step)
    lendiff_feature_path = '../feature_{}/lendiff_feature.txt'.format(step)
    lendiffrate_feature_path = '../feature_{}/lendiffrate_feature.txt'.format(step)
    tfidf_feature_path = '../feature_{}/tfidf_feature.txt'.format(step)
    ngramDistance_feature_path = '../feature_{}/ngramDistance_feature_feature.txt'.format(step)
    powerful_word_dside_feature_path = '../feature_{}/powerful_word_dside_feature.txt'.format(step)
    powerful_word_oside_feature_path = '../feature_{}/powerful_word_oside_feature.txt'.format(step)
    PowerfulWordDoubleSideRate_feature_path = '../feature_{}/PowerfulWordDoubleSideRate_feature.txt'.format(step)
    PowerfulWordOneSideRate_feautre_path = '../feature_{}/PowerfulWordOneSideRate_feautre.txt'.format(step)
    dul_num_path = '../feature_{}/dul_num.txt'.format(step)



    '''
    load feature
    '''
    powerful_word_dside_feature = DataUtil.load_matrix(powerful_word_dside_feature_path)
    PowerfulWordDoubleSideRate_feature = DataUtil.load_matrix(PowerfulWordDoubleSideRate_feature_path)
    no_feature = DataUtil.load_matrix(no_feature_path)
    # no_feature_min_max_transfer= preprocessing.MinMaxScaler()
    # no_feature_train_minmax = no_feature_min_max_transfer.fit_transform(no_feature)
    word_match_feature = DataUtil.load_matrix(word_match_path)
    ngramDistance_feature = DataUtil.load_matrix(ngramDistance_feature_path)
    tf_idf_word_share_feature = DataUtil.load_matrix(tf_idf_word_share_feature_path)

    dul_num = DataUtil.load_matrix(dul_num_path)
    dul_num = preprocessing.scale(dul_num)
    NgramDiceDistance_feature = DataUtil.load_matrix(NgramDiceDistance_feature_path)
    NgramJaccardCoef_feature = DataUtil.load_matrix(NgramJaccardCoef_feature_path)
    # Distance_feature = DataUtil.load_matrix(Distance_feature_path)
    len_feature = DataUtil.load_matrix(len_feature_path)
    lendiff_feature = DataUtil.load_matrix(lendiff_feature_path)
    lendiffrate_feature = DataUtil.load_matrix(lendiffrate_feature_path)
    tfidf_feature = DataUtil.load_matrix(tfidf_feature_path)
    tfidf_feature = np.nan_to_num(tfidf_feature)
    if step == 'train':
        cut_index = 3
    else:
        cut_index = 2
    train_distance_feature = pd.read_csv('../feature_{}/w2vec_features_scale.csv'.format(step),encoding='gbk')
    train_distance_feature = train_distance_feature.fillna(value=0)
    train_distance_feature = train_distance_feature.iloc[:,cut_index:]
    train_distance_feature.to_csv('../feature_{}/feature_deepnet.csv'.format(step),index=False)
    comb_feature = pd.read_csv('../feature_{}/comb.csv'.format(step),encoding='gbk')
    print('comb',comb_feature.shape)
    print('distance_featrue',train_distance_feature.shape)
    feature = np.concatenate((comb_feature,ngramDistance_feature,lendiffrate_feature,lendiff_feature,len_feature,
        tfidf_feature,powerful_word_dside_feature,NgramJaccardCoef_feature,NgramDiceDistance_feature,dul_num,
                              no_feature,word_match_feature,tf_idf_word_share_feature),axis=1)
    DataUtil.save_matrix('../feature_{}/feature.txt'.format(step),feature,'w')
def train_xgb():
    #
    import xgboost as xgb

    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 4
    feature1 = DataUtil.load_matrix('../feature_train/feature_min_max.txt')
    feature2 = pd.read_csv('../feature_train/feature_deepnet.csv').values
    print(feature1.shape)
    feature = np.concatenate([feature1,feature2],axis=1)
    print(feature.shape)
    label = np.load('../data/train_label.npy')
    x_train,x_valid,y_train,y_valid = train_test_split(feature,label)
    feature1_test = DataUtil.load_matrix('../feature_test/feature_min_max.txt')
    feature2_test = pd.read_csv('../feature_test/feature_deepnet.csv').values
    feature_test = np.concatenate([feature1_test,feature2_test],axis=1)
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    d_test = xgb.DMatrix(feature_test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    bst = xgb.train(params, d_train, 5000, watchlist, early_stopping_rounds=50, verbose_eval=10)
    pd.DataFrame(bst.predict(d_test)).to_csv('../result/result.csv')
if __name__ == '__main__':
    #
    save_feature('train')
    save_feature('test')
    train_feature = DataUtil.load_matrix('../feature_train/feature.txt')
    test_feature = DataUtil.load_matrix('../feature_test/feature.txt')
    print(train_feature.shape)
    feature = np.concatenate([train_feature,test_feature],axis=0)
    scale_transfer = preprocessing.StandardScaler()
    scale_transfer_fit = scale_transfer.fit(feature)
    train_feature_min_max = scale_transfer_fit.transform(train_feature)
    test_feature_min_max = scale_transfer_fit.transform(test_feature)
    DataUtil.save_matrix('../feature_train/feature_min_max.txt',train_feature_min_max,'w')
    DataUtil.save_matrix('../feature_test/feature_min_max.txt',test_feature_min_max,'w')