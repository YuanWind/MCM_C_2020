from sklearn.decomposition import pca
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from textblob import TextBlob
# blob = TextBlob ("text")
# print(blob.sentiment.polarity)
# out_put = emotion_eng.getMoodValue("great")#out_put['all_value']
# all_low=hair_dryer[(hair_dryer['star_rating']<2) & (hair_dryer['review_body']<-0.6)]
# all_high=hair_dryer[(hair_dryer['star_rating']>3) & (hair_dryer['review_body']>0.2)]
# all_mid=hair_dryer[(hair_dryer['star_rating']>=2) & (hair_dryer['star_rating']<=3) & (0.2>=hair_dryer['review_body']) & (hair_dryer['review_body']>=-0.6)]
# not_pair=hair_dryer[((hair_dryer['star_rating']<2) & (hair_dryer['review_body']>0.8)) | ((microwave['star_rating']==5) & (hair_dryer['review_body']<-0.6))]
#
# a=all_low.count()['star_rating']
# b=all_high.count()['star_rating']
# c=all_mid.count()['star_rating']
# d=hair_dryer.count()['star_rating']
# e=not_pair.count()['star_rating']
# print(a,b,c,e,d-a-b-c)
# print()
# all_low=microwave[(microwave['star_rating']<2) & (microwave['review_body']<-0.6)]
# all_high=microwave[(microwave['star_rating']>3) & (microwave['review_body']>0.2)]
# all_mid=microwave[(microwave['star_rating']>=2) & (microwave['star_rating']<=3) & (0.2>=microwave['review_body']) & (microwave['review_body']>=-0.6)]
# not_pair=microwave[((microwave['star_rating']<2) & (microwave['review_body']>0.8)) | ((microwave['star_rating']==5) & (microwave['review_body']<-0.6))]
#
# a=all_low.count()['star_rating']
# b=all_high.count()['star_rating']
# c=all_mid.count()['star_rating']
# d=microwave.count()['star_rating']
# e=not_pair.count()['star_rating']
# print(a,b,c,e,d-a-b-c)
# print()
# all_low=pacifier[(pacifier['star_rating']<2) & (pacifier['review_body']<-0.6)]
# all_high=pacifier[(pacifier['star_rating']>3) & (pacifier['review_body']>0.2)]
# all_mid=pacifier[(pacifier['star_rating']>=2) & (pacifier['star_rating']<=3) & (0.2>=pacifier['review_body']) & (pacifier['review_body']>=-0.6)]
#
# not_pair=pacifier[((pacifier['star_rating']<2) & (pacifier['review_body']>0.8)) | ((pacifier['star_rating']==5) & (pacifier['review_body']<-0.6))]
# a=all_low.count()['star_rating']
# b=all_high.count()['star_rating']
# c=all_mid.count()['star_rating']
# d=pacifier.count()['star_rating']
# e=not_pair.count()['star_rating']
# print(a,b,c,e,d-a-b-c)
hair_dryer=pd.read_csv('../Data/new_hair_dryer.csv',encoding='utf-8',index_col=0)
microwave=pd.read_csv('../Data/new_microwave.csv',encoding='utf-8',index_col=0)
pacifier=pd.read_csv('../Data/new_pacifier.csv',encoding='utf-8',index_col=0)
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
hair_dryer=gen_rate(hair_dryer)
microwave=gen_rate(microwave)
pacifier=gen_rate(pacifier)

def anylisis(data):
    all_low=data[(data['star_rating']<2) & (data['review_body']<-0.6)]
    all_high=data[(data['star_rating']>3) & (data['review_body']>0.2)]
    all_mid=data[(data['star_rating']>=2) & (data['star_rating']<=3) & (0.2>=data['review_body']) & (data['review_body']>=-0.6)]
    # not_pair 表示好评却1星或5星却差评
    not_pair=data[((data['star_rating']==1) & (data['review_body']>0.6)) | ((data['star_rating']==5) & (data['review_body']<-0.6))]
    a=all_low.count()['star_rating']
    b=all_high.count()['star_rating']
    c=all_mid.count()['star_rating']
    d=data.count()['star_rating']
    e=not_pair.count()['star_rating']
    # print(a,b,c,e,d-a-b-c)
    # print('好评却1星或5星却差评数量:',e)
    return not_pair.index.values
abnormal_product={}
abnormal_product['hair_dryer']=(list(anylisis(hair_dryer)))#8
abnormal_product['microwave']=(list(anylisis(microwave)))#3
abnormal_product['pacifier']=(list(anylisis(pacifier)))#18
print(abnormal_product)
def scaler(X):
    """
    注：这里的归一化是按照列进行的。也就是把每个特征都标准化，就是去除了单位的影响。
    """
    min_max_scaler = MinMaxScaler ()
    x_train= min_max_scaler.fit_transform (X)
    x=pd.DataFrame(x_train,columns=X.columns.values)
    return x
def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 标准化
    x = x.apply (lambda x: ((x - np.min (x)) / (np.max (x) - np.min (x))))

    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log (rows)

    lnf = [[None] * cols for i in range (rows)]

    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = np.array (x)
    lnf = [[None] * cols for i in range (rows)]
    lnf = np.array (lnf)
    for i in range (0, rows):
        for j in range (0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum (axis=0)[j]
                lnfij = math.log (p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame (lnf)
    E = lnf

    # 计算冗余度
    d = 1 - E.sum (axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range (cols)]
    for j in range (0, cols):
        wj = d[j] / sum (d)
        w[j] = wj
        # 计算各样本的综合得分,用最原始的数据

    w = pd.DataFrame (w)
    return w
def get_eval(prod,data):

    data=data[~data['product_id'].isin(abnormal_product[prod])]
    x=data[['star_rating','review_body']]
    # x=scaler(x)
    w = cal_weight (x)  # 调用cal_weight
    w.index = x.columns
    w.columns = ['weight']
    wei={'star_rating':w.loc['star_rating','weight'],'review_body':w.loc['review_body','weight']}
    return wei
# wei=get_eval('hair_dryer',hair_dryer) #{'star_rating': 0.8529774897515476, 'review_body': 0.1470225102484523}
# print(wei)
def gen_score(prod,data):
    x=data['star_rating'].values
    y=data['review_body'].values
    wei = get_eval (prod, data)
    score=np.array(x)*wei['star_rating']+np.array(y)*wei['review_body']
    data['score']=score
    return data

def fig(prod, D):
    data=gen_score(prod,D)[['review_date','year','month','score']]
    # print(data.describe())
    good=data[(data['score']>4) & (data['year']>2009)].groupby(['year','month']).count()['score']
    bad=data[(data['score']<1) & (data['year']>2009)].groupby(['year','month']).count()['score']
    all_of=data[(data['year']>2009)].groupby(['year','month']).count()['review_date']
    good=pd.DataFrame(good,index=good.index.values)
    bad=pd.DataFrame(bad,index=bad.index.values)
    all_of=pd.DataFrame(all_of,index=all_of.index.values)
    bad.rename(columns={'score':'score_bad'},inplace=True)
    x=[str(i[0])+'/'+str(i[1]) for i in good.index.values]
    good['time']=x
    x=[str(i[0])+'/'+str(i[1]) for i in bad.index.values]
    bad['time']=x
    x=[str(i[0])+'/'+str(i[1]) for i in all_of.index.values]
    all_of['time']=x
    all=pd.merge(good,bad,how='left')
    all=pd.merge(all,all_of,how='left')
    all.fillna(0)
    fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
    plt.plot(all['time'].values,all['score'].values/all['review_date'].values)
    # plt.show()
    plt.plot(all['time'].values,all['score_bad'].values/all['review_date'].values)
    # plt.plot(all['time'].values,all['review_date'].values)
    plt.legend(['good','bad'],loc = 'best')

    plt.xticks (size='small', rotation=90, fontsize=13)
    plt.show()

# fig('hair_dryer',hair_dryer)
# fig('microwave',microwave)
# fig('pacifier',pacifier)

def classify(prod,data):
    data=gen_score(prod, data)
    cols_x = ['helpful_votes', 'total_votes', 'verified_purchase', 'review_body', 'review_date', 'month', 'rate','score']
    x=data[cols_x]
    scores=data['star_rating'].values
    y=[]
    for score in scores:
        if score>=4:
            y.append(1)
        else:
            y.append(0)
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    ss = StandardScaler ()
    X_train = ss.fit_transform (X_train)
    X_test = ss.fit_transform (X_test)
    lr = LogisticRegression()
    lr.fit (X_train, y_train)
    lr_y_predict = lr.predict (X_test)
    print(lr_y_predict)
    print ('Accuracy of LR Classifier:', lr.score (X_test, y_test))

    print()


classify('hair_dryer',hair_dryer)
print()






