import pandas as pd
import matplotlib.pyplot as plt

def null_process():
    hair_dryer=pd.read_csv('../Data/hair_dryer.tsv',sep='\t',encoding='utf-8')
    microwave=pd.read_csv('../Data/microwave.tsv',sep='\t',encoding='utf-8')
    pacifier=pd.read_csv('../Data/pacifier.tsv',sep='\t',encoding='utf-8')
    # print(hair_dryer.head())
    for column in hair_dryer.columns:
        if len(set(hair_dryer[column]))==1:
            print('hair_dryer:',column)
        if len(set(microwave[column]))==1:
            print('microwave:',column)
        if len(set(pacifier[column]))==1:
            print('pacifier:',column)
    print(set(pacifier['product_category']))
    print(set(pacifier['marketplace']))
    print(set(microwave['product_category']))
    print(set(microwave['marketplace']))
    #'product_category', 'marketplace' 每一列的值都相同, 直接删掉

    del hair_dryer['product_category']
    del hair_dryer['marketplace']
    del pacifier['product_category']
    del pacifier['marketplace']
    del microwave['product_category']
    del microwave['marketplace']

    tmp1 = pacifier[pacifier['product_id'] == 'b0042i2bwg']
    print (tmp1)
    tmp2 = pacifier[pacifier['product_id'] == 'b00db5f114']
    print (tmp2)
    dic1 = {}
    for idx in hair_dryer.index:
        i = hair_dryer.loc[idx, 'product_id']
        j = hair_dryer.loc[idx, 'product_parent']
        if i not in dic1:
            dic1[i] = [j]
        else:
            dic1[i].append (j)
    for i in dic1:
        if len (set (dic1[i])) != 1:
            print ('hair_dryer')
    dic2 = {}
    for idx in microwave.index:
        i = microwave.loc[idx, 'product_id']
        j = microwave.loc[idx, 'product_parent']
        if i not in dic2:
            dic2[i] = [j]
        else:
            dic2[i].append (j)
    for i in dic2:
        if len (set (dic2[i])) != 1:
            print ('microwave')
    dic3 = {}
    for idx in pacifier.index:
        i = pacifier.loc[idx, 'product_id']
        j = pacifier.loc[idx, 'product_parent']
        if i not in dic3:
            dic3[i] = [j]
        else:
            dic3[i].append (j)
    for i in dic3:
        if len (set (dic3[i])) != 1:
            print (i, dic3[i])
            print ('pacifier')
    # 以上发现除了pacifier里边的两个异常值例外, 其他所有数据, 只要product_id一样, product_parent也一样故只保留一个字段
    del hair_dryer['product_parent']
    del microwave['product_parent']
    del pacifier['product_parent']

    print(hair_dryer['product_id'].count()) #11470
    print(microwave['product_id'].count())#1615
    print(pacifier['product_id'].count())#18939
    hair_dryer=hair_dryer.dropna()
    microwave=microwave.dropna()
    pacifier=pacifier.dropna()
    print(hair_dryer['product_id'].count())#11468
    print(microwave['product_id'].count())#1615
    print(pacifier['product_id'].count())#18937
    # 只有四行数据有缺失值, 不影响整体, 故直接删除之后重新保存
    hair_dryer.to_csv('../Data/hair_dryer.csv',encoding='utf-8',index=None)
    microwave.to_csv('../Data/microwave.csv',encoding='utf-8',index=None)
    pacifier.to_csv('../Data/pacifier.csv',encoding='utf-8',index=None)

    # 打印有缺失值的行
    # print(hair_dryer[hair_dryer.isnull().values==True])
    # print(microwave[microwave.isnull().values==True])
    # print(pacifier[pacifier.isnull().values==True])
# null_process()
hair_dryer=pd.read_csv('../Data/hair_dryer.csv',encoding='utf-8')
microwave=pd.read_csv('../Data/microwave.csv',encoding='utf-8')
pacifier=pd.read_csv('../Data/pacifier.csv',encoding='utf-8')
def fig_star_rating_count():
    tmp1=hair_dryer.groupby(by='star_rating').count()['customer_id']
    plt.subplot(221)
    plt.bar(tmp1.index.values,tmp1.values)
    plt.ylim(0,8000)
    for a, b in zip(tmp1.index.values, tmp1.values):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    plt.title('hair_dryer')

    tmp2=microwave.groupby(by='star_rating').count()['customer_id']
    plt.subplot(222)
    plt.ylim(0,800)
    plt.bar(tmp2.index.values,tmp2.values)
    for a, b in zip(tmp2.index.values, tmp2.values):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    plt.title('microwave')

    tmp3=pacifier.groupby(by='star_rating').count()['customer_id']
    plt.subplot(212)
    plt.ylim(0,14000)
    plt.bar(tmp3.index.values,tmp3.values)
    for a, b in zip(tmp3.index.values, tmp3.values):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    plt.title('pacifier')
    plt.show()
# fig_star_rating_count()

# test1=hair_dryer[hair_dryer['product_id']=='B003V264WW']
# print()

# print(hair_dryer.info())
# print(microwave.info())
# print(pacifier.info())
#
# print(hair_dryer.describe())
# print(microwave.describe())
# print(pacifier.describe())
# # 以上结论: 我们需要对数据取样, 因为要想准确显示评价真实性, 要结合helpful_votes/verified_purchase两个指标来看, 需要排除一些没有意义的评价
# print(hair_dryer[hair_dryer['verified_purchase']=='Y'].count())

print()
