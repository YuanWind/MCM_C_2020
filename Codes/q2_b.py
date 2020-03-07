from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import math
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
data=gen_score('hair_dryer',hair_dryer)[['review_date','year','month','score']]
# print(data.describe())
print(data[data['score']>4].count())
print(data[data['score']<2].count())






