import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
label = np.load('../data/train_label.npy').astype(np.int64)
esim1_best = pd.read_csv('./data/submit_clean_stops_number_punciton_07_30_07_37.txt',names=['score'])
attention_best = pd.read_csv('./data/submit_attention_07_30_17_48.txt',names=['score'])

shen_best = pd.read_csv('./data/shen.txt',names=['score'])
# print(attention_best.describe())
print(shen_best.describe())
print('best done')
attention1_test = pd.read_csv('./data/attention1_test.txt',names=['score'])
attention1_train = pd.read_csv('./data/attention1_train.txt',names=['score'])

esim1_test = pd.read_csv('./data/esim1_test.txt',names=['score'])
esim1_train = pd.read_csv('./data/esim1_train.txt',names=['score'])

lstm1_test = pd.read_csv('./data/lstm1_test.txt',names=['score'])
lstm1_train = pd.read_csv('./data/lstm1_train.txt',names=['score'])

deepnet1_test = pd.read_csv('./data/deepnet1_test.txt',names=['score'])
deepnet1_train = pd.read_csv('./data/deepnet1_train.txt',names=['score'])

X_train = np.hstack([esim1_train,attention1_train,lstm1_train,deepnet1_train])
X_test = np.hstack([esim1_test,attention1_test,lstm1_test,deepnet1_test])
print(X_test.shape)
print ('X_train shape',X_train.shape)
print('y_shape',label.shape)

te_pred=np.zeros(X_train.shape[0])
cnt=0
test_result = np.zeros(len(X_test))
for k,(ind_tr,ind_te) in enumerate(StratifiedKFold(label,random_state=27,n_folds=5)):
    clf = xgb.XGBClassifier(
            n_estimators=3000,
            learning_rate = 0.093/3,
            max_depth=8,
            colsample_bytree = 0.54,
            gamma = 0.3,
            reg_lambda=0,
            min_child_weight=13,
            seed = 1024,
            )
    train = X_train[ind_tr]
    test = X_train[ind_te]
    train_y = label[ind_tr]
    test_y = label[ind_te]
    clf.fit(train,train_y,eval_set=[(test,test_y)],eval_metric='logloss',early_stopping_rounds=100)
    test_pred = clf.predict_proba(X_test)[:,-1].flatten()
    print(test_pred)
    test_result += test_pred
    print (log_loss(test_y,clf.predict_proba(test)[:,-1].flatten()),log_loss(train_y,clf.predict_proba(train)[:,-1].flatten()))
    print(clf.predict_proba(test)[:,-1].flatten().mean())
    te_pred[ind_te]=clf.predict_proba(test)[:,-1].flatten()
    print('end fold:{}'.format(cnt))
    cnt+=1
result_df = pd.DataFrame({"score":test_result/5})
train_df = pd.DataFrame({"xgb1":te_pred})
result_df.to_csv('./data/xgb_stack2_test.txt',index=False,header=False)
train_df.to_csv('./data/xgb_stack2_train.csv',index=False,header=False)
submit = (result_df['score'] + attention_best['score']+esim1_best['score']+shen_best['score'])/4
submit.to_csv('./data/submit_0731.txt',index=False,header=False)
print(submit.describe())
print(train_df.describe())
#
graph_df = pd.read_csv('./data/test_graph_feature_update.csv')
result_8th  = pd.read_table('./data/submit_0731.txt',names=['score'])
