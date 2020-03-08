import collections
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jieba.analyse
import seaborn as sns
from tqdm import tqdm

from my_util import pre_process
import pandas as pd
import wordcloud
# 基于TF - IDF：jieba.analyse.extract_tags (sentence, topK=20, withWeight=False, allowPOS=())
# 基于TextRank：jieba.analyse.textrank (sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
hair_dryer=pd.read_csv('../Data/hair_dryer.csv',encoding='utf-8')
microwave=pd.read_csv('../Data/microwave.csv',encoding='utf-8')
pacifier=pd.read_csv('../Data/pacifier.csv',encoding='utf-8')
hair_dryer = hair_dryer.dropna ()
microwave = microwave.dropna ()
pacifier = pacifier.dropna ()
def try1():
    def gen_star_sent(n):
        tmp1=hair_dryer[hair_dryer['star_rating']==n]['review_body']
        tmp2=microwave[microwave['star_rating']==n]['review_body']
        tmp3=pacifier[pacifier['star_rating']==n]['review_body']
        star_str=''
        for i in tqdm(tmp1.values):
            for j in pre_process(i):
                star_str=star_str+' '+j
        for i in tqdm(tmp2.values):
            for j in pre_process (i):
                star_str = star_str + ' ' + j
        for i in tqdm(tmp3.values):
            for j in pre_process (i):
                star_str = star_str + ' ' + j
        print()
        return star_str
    one_star_sen=gen_star_sent(1)
    two_star_sen=gen_star_sent(2)
    three_star_sen=gen_star_sent(3)
    four_star_sen=gen_star_sent(4)
    five_star_sen=gen_star_sent(5)
    # one_star_sen=hair_dryer[hair_dryer['star_rating']==1]['review_body']+microwave[microwave['star_rating']==1]['review_body']+pacifier[pacifier['star_rating']==1]['review_body']
    # keywords=jieba.analyse.extract_tags(one_star_sen, topK=20, withWeight=False, allowPOS=())
    # print(keywords)
    w = wordcloud.WordCloud(max_words=50)
    w.generate(one_star_sen)
    w.to_file('output1.png')

    # keywords=jieba.analyse.extract_tags(two_star_sen, topK=20, withWeight=False, allowPOS=())
    # print(keywords)
    w = wordcloud.WordCloud(max_words=50)
    w.generate(two_star_sen)
    w.to_file('output2.png')

    keywords=jieba.analyse.extract_tags(three_star_sen, topK=20, withWeight=False, allowPOS=())
    print(keywords)
    w = wordcloud.WordCloud(max_words=50)
    w.generate(three_star_sen)
    w.to_file('output3.png')

    # keywords=jieba.analyse.extract_tags(four_star_sen, topK=20, withWeight=False, allowPOS=())
    # print(keywords)
    w = wordcloud.WordCloud(max_words=50)
    w.generate(four_star_sen)
    w.to_file('output4.png')

    # keywords=jieba.analyse.extract_tags(five_star_sen, topK=20, withWeight=False, allowPOS=())
    # print(keywords)
    w = wordcloud.WordCloud(max_words=50)
    w.generate(five_star_sen)
    w.to_file('output5.png')

def try2():
    # words_list=set()
    # with open('emotion_dict/words_list.txt','r',encoding='utf-8') as f:
    #     for line in f:
    #         words_list.add(line.replace('\n',''))
    # def gen_star_sent(n):
    #     tmp1 = hair_dryer[hair_dryer['star_rating'] == n]['review_body']
    #     tmp2 = microwave[microwave['star_rating'] == n]['review_body']
    #     tmp3 = pacifier[pacifier['star_rating'] == n]['review_body']
    #
    #     star_str = ''
    #     for i in tqdm(tmp1.values):
    #         for j in pre_process(i):
    #             if j in words_list:
    #                 star_str = star_str + ' ' + j
    #     for i in tqdm(tmp2.values):
    #         for j in pre_process (i):
    #             if j in words_list:
    #                 star_str = star_str + ' ' + j
    #     for i in tqdm(tmp3.values):
    #         for j in pre_process (i):
    #             if j in words_list:
    #                 star_str = star_str + ' ' + j
    #     print()
    #
    #     return star_str
    #
    # one_star_sen = gen_star_sent (1)
    # two_star_sen = gen_star_sent (2)
    # three_star_sen = gen_star_sent (3)
    # four_star_sen = gen_star_sent (4)
    # five_star_sen = gen_star_sent (5)
    # star_sent = {}
    # star_sent['one'] = one_star_sen
    # star_sent['two'] = two_star_sen
    # star_sent['three'] = three_star_sen
    # star_sent['four'] = four_star_sen
    # star_sent['five'] = five_star_sen
    # pickle.dump (star_sent, open ('star_sent_cloud.pkl', 'wb'))
    star_sent=pickle.load(open('star_sent_cloud.pkl','rb'))

    w = wordcloud.WordCloud (max_words=50)
    w.generate (star_sent['one'])
    w.to_file ('output1.png')


    w = wordcloud.WordCloud (max_words=50)
    w.generate (star_sent['two'])
    w.to_file ('output2.png')

    w = wordcloud.WordCloud (max_words=50)
    w.generate (star_sent['three'])
    w.to_file ('output3.png')

    w = wordcloud.WordCloud (max_words=50)
    w.generate (star_sent['four'])
    w.to_file ('output4.png')

    w = wordcloud.WordCloud (max_words=50)
    w.generate (star_sent['five'])
    w.to_file ('output5.png')

def try3(num):
    words_list = set ()
    with open ('emotion_dict/words_list.txt', 'r', encoding='utf-8') as f:
        for line in f:
            words_list.add (line.replace ('\n', ''))

    def gen_star_sent1(n):
        tmp1 = hair_dryer[hair_dryer['star_rating'] == n]['review_body']
        tmp2 = microwave[microwave['star_rating'] == n]['review_body']
        tmp3 = pacifier[pacifier['star_rating'] == n]['review_body']

        star_str = []
        for i in tqdm (tmp1.values):
            for j in pre_process (i):
                if j in words_list:
                    star_str .append(j)
        for i in tqdm (tmp2.values):
            for j in pre_process (i):
                if j in words_list:
                    star_str .append(j)
        for i in tqdm (tmp3.values):
            for j in pre_process (i):
                if j in words_list:
                    star_str .append(j)
        print()

        return star_str

    # one_star_sen = gen_star_sent1 (1)
    # two_star_sen = gen_star_sent1 (2)
    # three_star_sen = gen_star_sent1 (3)
    # four_star_sen = gen_star_sent1 (4)
    five_star_sen = gen_star_sent1 (5)
    # star_sent={}
    # star_sent['one']=one_star_sen
    # star_sent['two']=two_star_sen
    # star_sent['three']=three_star_sen
    # star_sent['four']=four_star_sen
    # star_sent['five']=five_star_sen
    # pickle.dump(star_sent,open('star_sent_count.pkl','wb'))
    star_sent=pickle.load(open('star_sent_count.pkl','rb'))
    words_set=[]
    words_dict={}

    word_counts = collections.Counter (star_sent['five'])  # 对分词做词频统计
    word_counts_top20 = word_counts.most_common (num)  # 获取前20最高频的词
    # print (word_counts_top20)  # 输出检查
    word_counts_top20 = dict (word_counts_top20)
    words_set += list (word_counts_top20.keys ())
    words_dict['5'] = word_counts_top20
    y5 = list(word_counts_top20.values())
    x5 = [5 for _ in range (len (y5))]

    word_counts = collections.Counter (star_sent['four'])  # 对分词做词频统计
    word_counts_top20 = word_counts.most_common (num)  # 获取前20最高频的词
    # print (word_counts_top20)  # 输出检查
    word_counts_top20 = dict (word_counts_top20)
    words_set += list (word_counts_top20.keys ())
    words_dict['4'] = word_counts_top20
    y4 = list(word_counts_top20.values())
    x4 = [4 for _ in range (len (y4))]

    word_counts = collections.Counter (star_sent['three'])  # 对分词做词频统计
    word_counts_top20 = word_counts.most_common (num)  # 获取前20最高频的词
    # print (word_counts_top20)  # 输出检查
    word_counts_top20 = dict (word_counts_top20)
    words_set += list (word_counts_top20.keys ())
    words_dict['3'] = word_counts_top20
    y3 = list(word_counts_top20.values())
    x3 = [3 for _ in range (len (y3))]

    word_counts = collections.Counter (star_sent['two'])  # 对分词做词频统计
    word_counts_top20 = word_counts.most_common (num)  # 获取前20最高频的词
    # print (word_counts_top20)  # 输出检查
    word_counts_top20 = dict (word_counts_top20)
    words_set += list (word_counts_top20.keys ())
    words_dict['2'] = word_counts_top20
    y2 = list(word_counts_top20.values())
    x2 = [2 for _ in range (len (y2))]

    word_counts = collections.Counter (star_sent['one'])  # 对分词做词频统计
    word_counts_top20 = word_counts.most_common (num)  # 获取前20最高频的词
    word_counts_top20=dict(word_counts_top20)
    words_set+=list(word_counts_top20.keys())
    words_dict['1']=word_counts_top20
    # print (word_counts_top20)  # 输出检查
    y1 = list(word_counts_top20.values())
    x1=[1 for _ in range(len(y1))]

    words_set=set(words_set)
    sorted(words_set)
    dic={'1':[],'2':[],'3':[],'4':[],'5':[]}
    for i in words_set:
        for j in range(1,6):
            if i in words_dict[str(j)].keys():
                dic[str(j)].append(words_dict[str(j)][i])
            else:
                dic[str(j)].append(0)

    data=pd.DataFrame(dic,index=words_set)
    data.to_csv('../Data/word_count'+str(num)+'.csv',encoding='utf-8')
    cmap = sns.cubehelix_palette (start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap (data,linewidths = 0.05, vmax=5000, vmin=50, cmap=cmap)
    plt.show()
    plt.savefig('words_hot'+str(num)+'.png')
    # x=x1+x2+x3+x4+x5
    # y=y1+y2+y3+y4+y5
    # fig,ax=plt.subplots()
    # ax.scatter(x,y,c='r')
    # plt.show()


try3(10)
# try2()