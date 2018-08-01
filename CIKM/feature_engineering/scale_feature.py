import pandas as pd
from sklearn import preprocessing


train_feature = pd.read_csv('../feature_train/w2vec_distance_features.csv',encoding='gbk')
test_feature = pd.read_csv('../feature_test/w2vec_distance_features.csv',encoding='gbk')

feature = pd.concat([train_feature,test_feature],axis=0)
print(feature)
# feature['wmd'] = feature['wmd'].replace('inf',5)
# feature['norm_wmd'] = feature['norm_wmd'].replace('inf',1.5)
# feature_wmd_distance = feature['wmd']
# feature_normwmd_distance = feature['norm_wmd']
# wmd_scale_transfer = preprocessing.MinMaxScaler()
# wmd_scale_transfer_fit = wmd_scale_transfer.fit(feature['wmd'])
#
# norm_wmd_scale_transfer = preprocessing.MinMaxScaler()
# norm_wmd_scale_transfer_fit = norm_wmd_scale_transfer.fit(feature['norm_wmd'])


def scale_data(step):
    feature = pd.read_csv('../feature_{}/w2vec_distance_features.csv'.format(step),encoding='gbk')
    # feature['wmd'] = feature['wmd'].replace('inf',5)
    # feature['wmd'] = wmd_scale_transfer_fit.fit_transform(feature['wmd'].values)
    #
    # feature['norm_wmd'] = feature['norm_wmd'].replace('inf',1.5)
    # feature['norm_wmd'] = norm_wmd_scale_transfer_fit.fit_transform(feature['norm_wmd'].values)
    # print(feature['norm_wmd'].describe())
    feature['fuzz_qratio'] = feature['fuzz_qratio']/100
    feature['fuzz_WRatio'] = feature['fuzz_WRatio']/100
    feature['fuzz_partial_ratio'] = feature['fuzz_partial_ratio']/100
    feature['fuzz_partial_token_set_ratio'] = feature['fuzz_partial_token_set_ratio']/100
    feature['fuzz_partial_token_sort_ratio'] = feature['fuzz_partial_token_sort_ratio']/100
    feature['fuzz_token_set_ratio'] = feature['fuzz_token_set_ratio']/100
    feature['fuzz_token_sort_ratio'] = feature['fuzz_token_sort_ratio']/100
    del feature['cityblock_distance']
    feature.to_csv('../feature_{}/w2vec_features_scale.csv'.format(step),index=False)
scale_data('train')
scale_data('test')
