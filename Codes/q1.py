from textblob import TextBlob
#
# text = "five stars".replace('.','')
# blob = TextBlob (text)
# # 分句
# print ("blob对象")
# print (blob)
# print (blob.sentiment)
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
import numpy as np
cat_cols = ['customer_id', 'review_id', 'product_id', 'vine', 'verified_purchase']
def pre_process(prod,data):
    del data['review_headline']
    dtime = pd.to_datetime (data['review_date'])
    v = (dtime.values - np.datetime64 ('2000-01-01T08:00:00Z')) / np.timedelta64 (1, 'ms')
    data['review_date'] = v

    data[cat_cols].astype('category')
    def map_value(x):
        x_set=set(x.values)
        dic={}
        n=0
        for i in x_set:
            if i not in dic:
                dic[i]=n
                n+=1
        new_x=[]
        for i in x.values:
            new_x.append(dic[i])
        return new_x
    for col in cat_cols:
        data[col] = map_value(data[col])
    def get_sentment(col):
        new_col=[]
        # for i in tqdm(col):
        #     out_put = emotion_eng.getMoodValue(i)
        #     new_col.append(out_put['all_value'])
        for i in tqdm(col):
            out_put = TextBlob (i)
            new_col.append(out_put.sentiment.polarity)
        return new_col
    data['review_body']=get_sentment(data['review_body'])

    def anylisis(data):
        not_pair = data[((data['star_rating'] == 1) & (data['review_body'] > 0.6)) | ((data['star_rating'] == 5) & (data['review_body'] < -0.6))]
        return not_pair.index.values

    abnormal_product = {}
    abnormal_product[prod] = (list (anylisis (data)))  # 8
    # abnormal_product['microwave'] = (list (anylisis (microwave)))  # 3
    # abnormal_product['pacifier'] = (list (anylisis (pacifier)))  # 18
    data = data[~data['product_id'].isin (abnormal_product[prod])]
    return data
hair_dryer=pd.read_csv('../Data/hair_dryer.csv',encoding='utf-8')
hair_dryer=hair_dryer.dropna()
hair_dryer=pre_process('hair_dryer',hair_dryer)
hair_dryer.to_csv('../Data/new_hair_dryer.csv')

microwave=pd.read_csv('../Data/microwave.csv',encoding='utf-8')
microwave=microwave.dropna()
microwave=pre_process('microwave',microwave)
microwave.to_csv('../Data/new_microwave.csv')

pacifier=pd.read_csv('../Data/pacifier.csv',encoding='utf-8')
pacifier=pacifier.dropna()
pacifier=pre_process('pacifier',pacifier)
pacifier.to_csv('../Data/new_pacifier.csv')

def get_X_y(prod,data):


    cols_x=['customer_id', 'review_id', 'product_id', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase','review_body', 'review_date', 'year', 'month','rate']
    star=[]
    for i in data['star_rating']:
        if i<4:
            star.append(0)
        else:
            star.append(1)
    data['star_rating']=star
    X=data[cols_x]
    X[cat_cols]=X[cat_cols].astype('category')
    # X['customer_id']=X['customer_id'].astype('category')
    y=data['star_rating']

    # min_max_scaler = MinMaxScaler ()
    # X= min_max_scaler.fit_transform (X)
    # da=pd.DataFrame(X,columns=cols_x)
    return X,y
def gen_rate(data):
    tmp = data.groupby ('product_id').count ()['customer_id']
    sums = {}
    for i in tmp.index.values:
        sums[i] = tmp[i]
    rate = {}
    for i in sums:
        cnt = data[(data['product_id'] == i) & (data['star_rating'] < 4)].count ()[0]
        rate[i] = cnt / sums[i]
    rates = []
    for i in data['product_id'].values:
        rates.append (rate[i])
    data['rate'] = rates
    return data
hair_dryer=pd.read_csv('../Data/new_hair_dryer.csv',encoding='utf-8',index_col=0)
microwave=pd.read_csv('../Data/new_microwave.csv',encoding='utf-8',index_col=0)
pacifier=pd.read_csv('../Data/new_pacifier.csv',encoding='utf-8',index_col=0)
hair_dryer=gen_rate(hair_dryer)
microwave=gen_rate(microwave)
pacifier=gen_rate(pacifier)
def model1():
    X,y=get_X_y('hair_dryer',hair_dryer)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0,shuffle=True)
    print("Train data length:", len(X_train))
    print("Test data length:", len(X_test))
    print('开始训练!')

    # 训练 cv and train
    gbm = lgb.sklearn.LGBMClassifier(boosting_type='gbdt', num_leaves=64, max_depth=-1, learning_rate=0.09, n_estimators=10, max_bin=255, subsample_for_bin=200000, objective=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True)
    gbm.fit(X_train,y_train,sample_weight=None, init_score=None,
                eval_set=None, eval_names=None, eval_sample_weight=None,
                eval_class_weight=None, eval_init_score=None, eval_metric=None,
                early_stopping_rounds=None, verbose=True,
                feature_name='auto', categorical_feature='auto', callbacks=None)
    print ('Start predicting...')
    # 预测数据集
    y_pred = gbm.predict (X_test)
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    print('importance:',list(zip(X_train.columns.values,gbm.feature_importances_)))
    precision=precision_score(y_test, y_pred)
    recall=recall_score(y_test, y_pred)
    print(list(zip(y_test.values,y_pred)))
    print ('正确率：', precision)
    print ('召回率：', recall)
    print ('auc值：', roc_auc_score (y_test, y_pred))
    print ('F1值：', 2 * (precision * recall) / (precision + recall))
def model2():
    X, y = get_X_y ('microwave',microwave)
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=0, shuffle=True)
    print ("Train data length:", len (X_train))
    print ("Test data length:", len (X_test))
    print ('开始训练!')

    # 训练 cv and train
    gbm = lgb.sklearn.LGBMClassifier (boosting_type='gbdt', num_leaves=64, max_depth=-1, learning_rate=0.1,
                                      n_estimators=10, max_bin=255, subsample_for_bin=200000, objective=None,
                                      min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0,
                                      subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                                      random_state=None, n_jobs=-1, silent=True)
    gbm.fit (X_train, y_train, sample_weight=None, init_score=None,
             eval_set=None, eval_names=None, eval_sample_weight=None,
             eval_class_weight=None, eval_init_score=None, eval_metric=None,
             early_stopping_rounds=None, verbose=True,
             feature_name='auto', categorical_feature='auto', callbacks=None)
    print ('Start predicting...')
    # 预测数据集
    y_pred = gbm.predict (X_test)
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    print ('importance:', list (zip (X_train.columns.values, gbm.feature_importances_)))
    precision = precision_score (y_test, y_pred)
    recall = recall_score (y_test, y_pred)
    print (list (zip (y_test.values, y_pred)))
    print ('正确率：', precision)
    print ('召回率：', recall)
    print ('auc值：', roc_auc_score (y_test, y_pred))
    print ('F1值：', 2 * (precision * recall) / (precision + recall))
def model3():
    X, y = get_X_y ('pacifier',pacifier)
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=0, shuffle=True)
    print ("Train data length:", len (X_train))
    print ("Test data length:", len (X_test))
    print ('开始训练!')

    # 训练 cv and train
    gbm = lgb.sklearn.LGBMClassifier (boosting_type='gbdt', num_leaves=64, max_depth=-1, learning_rate=0.09,
                                      n_estimators=10, max_bin=255, subsample_for_bin=200000, objective=None,
                                      min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0,
                                      subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                                      random_state=None, n_jobs=-1, silent=True)
    gbm.fit (X_train, y_train, sample_weight=None, init_score=None,
             eval_set=None, eval_names=None, eval_sample_weight=None,
             eval_class_weight=None, eval_init_score=None, eval_metric=None,
             early_stopping_rounds=None, verbose=True,
             feature_name='auto', categorical_feature='auto', callbacks=None)
    print ('Start predicting...')
    # 预测数据集
    y_pred = gbm.predict (X_test)
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    print ('importance:', list (zip (X_train.columns.values, gbm.feature_importances_)))
    precision = precision_score (y_test, y_pred)
    recall = recall_score (y_test, y_pred)
    print (list (zip (y_test.values, y_pred)))
    print ('正确率：', precision)
    print ('召回率：', recall)
    print ('auc值：', roc_auc_score (y_test, y_pred))
    print ('F1值：', 2 * (precision * recall) / (precision + recall))
# hair_dryer
model1()
#microwave
model2()
#pacifier
model3()