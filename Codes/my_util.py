'''
    工具模块
'''
import jieba
from nltk import word_tokenize,pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from rake_nltk import Rake

import re

def tokenize(sentence):
    '''
        去除多余空白、分词、词性标注
    '''
    sentence = re.sub(r'\s+', ' ', sentence)
    token_words = word_tokenize(sentence)
    token_words = pos_tag(token_words)   
    return token_words

wordnet_lematizer = WordNetLemmatizer()
def stem(token_words):
    '''
        词形归一化
    '''
    words_lematizer = []
    for word, tag in token_words:
        word_lematizer =  wordnet_lematizer.lemmatize(word)
        if tag.startswith('NN'):
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
        elif tag.startswith('VB'): 
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='v')   # v代表动词
        elif tag.startswith('JJ'): 
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='a')   # a代表形容词
        elif tag.startswith('R'): 
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='r')   # r代表代词
        else: 
            word_lematizer =  wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    return words_lematizer


sr = set(stopwords.words('english'))
sr.update(['I','i','at','based','in','of','for','on','and','to','an','using','with','the'])
def delete_stopwords(token_words):
    '''
        去停用词
    '''
    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words

def is_number(s):
    '''
        判断字符串是否为数字
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

characters = [' ',',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','...','^','{','}']
characters = set(characters + list('[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～]+'))
def delete_characters(token_words):
    '''
        去除特殊字符、数字
    '''
    words_list = [word for word in token_words if word not in characters and not is_number(word)]
    return words_list

def to_lower(token_words):
    '''
        统一为小写
    '''
    words_lists = [x.lower() for x in token_words]
    return words_lists

def pre_process(text):
    '''
        文本预处理
    '''
    try:
        token_words = tokenize(text)
        token_words = stem(token_words)
        token_words = delete_stopwords(token_words)
        token_words = delete_characters(token_words)
        token_words = to_lower(token_words)
    except :
        print(text)
        exit(-1)
    return token_words

def get_words(text):
    '''
        文本预处理
    '''
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    abstract_list = jieba.analyse.extract_tags (token_words, topK=25)

    return abstract_list


