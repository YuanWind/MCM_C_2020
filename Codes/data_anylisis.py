import pandas as pd

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
    hair_dryer.to_csv('../Data/hair_dryer.csv',encoding='utf-8')
    microwave.to_csv('../Data/microwave.csv',encoding='utf-8')
    pacifier.to_csv('../Data/pacifier.csv',encoding='utf-8')

    # 打印有缺失值的行
    # print(hair_dryer[hair_dryer.isnull().values==True])
    # print(microwave[microwave.isnull().values==True])
    # print(pacifier[pacifier.isnull().values==True])

# print(hair_dryer.info())
# print(microwave.info())
# print(pacifier.info())

# print(hair_dryer.describe())
# print(microwave.describe())
# print(pacifier.describe())
# 以上结论: 我们需要对数据取样, 因为要想准确显示评价真实性, 要结合helpful_votes/verified_purchase两个指标来看, 需要排除一些没有意义的评价
# print(hair_dryer[hair_dryer['verified_purchase']=='Y'].count())
null_process()
print()
