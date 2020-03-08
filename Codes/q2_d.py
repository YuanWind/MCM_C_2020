import collections
import pickle

import jieba.analyse
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
    words_list=set()
    with open('emotion_dict/words_list.txt','r',encoding='utf-8') as f:
        for line in f:
            words_list.add(line.replace('\n',''))
    def gen_star_sent(n):
        tmp1 = hair_dryer[hair_dryer['star_rating'] == n]['review_body']
        tmp2 = microwave[microwave['star_rating'] == n]['review_body']
        tmp3 = pacifier[pacifier['star_rating'] == n]['review_body']

        star_str = ''
        for i in tqdm(tmp1.values):
            for j in pre_process(i):
                if j in words_list:
                    star_str = star_str + ' ' + j
        for i in tqdm(tmp2.values):
            for j in pre_process (i):
                if j in words_list:
                    star_str = star_str + ' ' + j
        for i in tqdm(tmp3.values):
            for j in pre_process (i):
                if j in words_list:
                    star_str = star_str + ' ' + j
        print()

        return star_str

    one_star_sen = gen_star_sent (1)
    two_star_sen = gen_star_sent (2)
    three_star_sen = gen_star_sent (3)
    four_star_sen = gen_star_sent (4)
    five_star_sen = gen_star_sent (5)
    star_sent = {}
    star_sent['one'] = one_star_sen
    star_sent['two'] = two_star_sen
    star_sent['three'] = three_star_sen
    star_sent['four'] = four_star_sen
    star_sent['five'] = five_star_sen
    pickle.dump (star_sent, open ('star_sent_cloud.pkl', 'wb', encoding='utf-8'))
    w = wordcloud.WordCloud (max_words=50)
    w.generate (one_star_sen)
    w.to_file ('output1.png')


    w = wordcloud.WordCloud (max_words=50)
    w.generate (two_star_sen)
    w.to_file ('output2.png')

    w = wordcloud.WordCloud (max_words=50)
    w.generate (three_star_sen)
    w.to_file ('output3.png')

    w = wordcloud.WordCloud (max_words=50)
    w.generate (four_star_sen)
    w.to_file ('output4.png')

    w = wordcloud.WordCloud (max_words=50)
    w.generate (five_star_sen)
    w.to_file ('output5.png')

def try3():
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

    one_star_sen = gen_star_sent1 (1)
    two_star_sen = gen_star_sent1 (2)
    three_star_sen = gen_star_sent1 (3)
    four_star_sen = gen_star_sent1 (4)
    five_star_sen = gen_star_sent1 (5)
    star_sent={}
    star_sent['one']=one_star_sen
    star_sent['two']=two_star_sen
    star_sent['three']=three_star_sen
    star_sent['four']=four_star_sen
    star_sent['five']=five_star_sen
    pickle.dump(star_sent,open('star_sent_count.pkl','wb',encoding='utf-8'))

    word_counts = collections.Counter (one_star_sen)  # 对分词做词频统计
    word_counts_top10 = word_counts.most_common (20)  # 获取前10最高频的词
    print (word_counts_top10)  # 输出检查

    word_counts = collections.Counter (one_star_sen)  # 对分词做词频统计
    word_counts_top10 = word_counts.most_common (20)  # 获取前10最高频的词
    print (word_counts_top10)  # 输出检查

    word_counts = collections.Counter (two_star_sen)  # 对分词做词频统计
    word_counts_top10 = word_counts.most_common (20)  # 获取前10最高频的词
    print (word_counts_top10)  # 输出检查

    word_counts = collections.Counter (three_star_sen)  # 对分词做词频统计
    word_counts_top10 = word_counts.most_common (20)  # 获取前10最高频的词
    print (word_counts_top10)  # 输出检查

    word_counts = collections.Counter (four_star_sen)  # 对分词做词频统计
    word_counts_top10 = word_counts.most_common (20)  # 获取前10最高频的词
    print (word_counts_top10)  # 输出检查

    word_counts = collections.Counter (five_star_sen)  # 对分词做词频统计
    word_counts_top10 = word_counts.most_common (20)  # 获取前10最高频的词
    print (word_counts_top10)  # 输出检查

try3()
try2()