import datetime
import math

import numpy as np
import pandas as pd
# --------------------------------------------------------------------------------
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def months(str1,str2):
    year1=datetime.datetime.strptime(str1[0:10],"%Y-%m-%d").year
    year2=datetime.datetime.strptime(str2[0:10],"%Y-%m-%d").year
    month1=datetime.datetime.strptime(str1[0:10],"%Y-%m-%d").month
    month2=datetime.datetime.strptime(str2[0:10],"%Y-%m-%d").month
    num=(year1-year2)*12+(month1-month2)
    return num
hair_dryer=pd.read_csv('../Data/hair_dryer.csv',encoding='utf-8')
microwave=pd.read_csv('../Data/microwave.csv',encoding='utf-8')
pacifier=pd.read_csv('../Data/pacifier.csv',encoding='utf-8')

def review_s(data):
    review_score=[]
    for i in data.index.values:
        C1=13.1056
        C2=350
        C3=20.04
        H=data.loc[i,'helpful_votes']
        T=data.loc[i,'total_votes']
        X=np.log(1750)/np.log(C2)
        Length=len(data.loc[i,'review_body'])
        # Recency=-2*(1/(np.exp(months('2020-03-07',hair_dryer.loc[i,'review_date']))))
        # try:
        Recency= 2 * 1 / (1 + np.exp((months('2015-11-01',data.loc[i,'review_date'])) / (30)))
        # except:
            # print(i)
            # print(hair_dryer.loc[i,'review_date'])
        if data.loc[i,'verified_purchase']=='Y':
            Trust=2*(sigmoid(0.1*data[data['review_id']==data.loc[i,'review_id']].count()['helpful_votes'])-0.5)
        else:
            Trust=sigmoid(0.1*data[data['review_id']==data.loc[i,'review_id']].count()['helpful_votes'])-0.5
        # Review=(H+C1+np.log(min(1750,Length)))/(np.log(350)*(T+C2))+(X/2*Trust+X/2*Recency)/(T+C2)
        Review=((H + C1 + np.log (min (Length, 1750)) / np.log (350) + X / 2 * Trust + X / 2 * Recency)) / ((T + C2))
        review_score.append(Review)
    print(max(review_score))
    print(min(review_score))
    return review_score
review_score1=review_s(hair_dryer)
review_score2=review_s(microwave)
review_score3=review_s(pacifier)

hair_dryer=pd.read_csv('../Data/new_hair_dryer.csv',encoding='utf-8',index_col=0)
microwave=pd.read_csv('../Data/new_microwave.csv',encoding='utf-8',index_col=0)
pacifier=pd.read_csv('../Data/new_pacifier.csv',encoding='utf-8',index_col=0)
hair_dryer['review_score']=(review_score1 - np.min (review_score1)) / (np.max (review_score1) - np.min (review_score1)) #(review_score1 - np.min (review_score1)) / (np.max (review_score1) - np.min (review_score1)
microwave['review_score']=(review_score2 - np.min (review_score2)) / (np.max (review_score2) - np.min (review_score2))
pacifier['review_score']=(review_score3 - np.min (review_score3)) / (np.max (review_score3) - np.min (review_score3))
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
    x=data[['review_body','review_score']]
    # x=scaler(x)
    w = cal_weight (x)  # 调用cal_weight
    w.index = x.columns
    w.columns = ['weight']
    wei={'review_body':w.loc['review_body','weight'],'review_score':w.loc['review_score','weight']}
    # print(wei)
    return wei

def gen_score(prod,data):
    x=data['review_body'].values
    y=data['review_score'].values
    wei = get_eval (prod, data)
    print(prod,wei)
    score=np.array(x)*wei['review_body']+np.array(y)*wei['review_score']
    data['score']=score
    return data
data1=gen_score('hair_dryer',hair_dryer)
data2=gen_score('microwave',microwave)
data3=gen_score('pacifier',pacifier)
print(data1.head())
print(data2.head())
print(data3.head())