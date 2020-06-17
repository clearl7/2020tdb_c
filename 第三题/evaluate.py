# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:12:37 2020

@author: acer
"""

import xlrd
import os
import codecs
import jieba.posseg as pseg
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd

def Stopword():
    os.chdir('D:/')
    stopword=[w.strip() for w in codecs.open("stopwords.txt",'r','utf-8').readlines()]
    return stopword

def corpus_segment(table,n,v):
    stopword=Stopword()
    cixingofword = []  # 储存分词后的词语对应的词性
    wordtocixing = []  # 储存分词后的词语
    cutcorpus=[0]*(n-1)
    for i in range(n-1):
        content=table.cell(i+1,v).value
        content1 = re.sub(r'[^\u4e00-\u9fa5]', "",content)#只保留中文
        content1 = pseg.cut(content1)
        cutcorpus[i] = ""
        for words,cixing in content1:
            if words.strip() not in stopword and len(words.strip())>1:
                cutcorpus[i] =cutcorpus[i]+words.strip()+' '
                cixingofword.append(cixing)
                wordtocixing.append(words)
    print("文本分词结束！")
    # 自己造一个{“词语”:“词性”}的字典，方便后续使用词性
    word2flagdict = {wordtocixing[i]:cixingofword[i] for i in range(len(wordtocixing))}
    return cutcorpus,word2flagdict
    

def new_weight(word,weight,word_dict):
    wordflagweight = [1 for i in range(len(word))]   #这个是词性系数，需要调整系数来看效果
    word2=[]
    for i in range(len(word)):
        if(word_dict[word[i]]=="n"):  # 名词重要一点，我们就给它1.2
            wordflagweight[i] = 1.25
            word2.append(word[i])
        elif(word_dict[word[i]]=="vn"):
            wordflagweight[i] = 1
            word2.append(word[i])
        elif(word_dict[word[i]]=="v"):
            wordflagweight[i] = 1.1
            word2.append(word[i])
        elif(word_dict[word[i]]=="ns"):
            wordflagweight[i] = 1
            word2.append(word[i])
        elif(word_dict[word[i]]=="m"):  # 只是举个例子，这种量词什么的直接去掉，省了一步停用词词典去除
            wordflagweight[i] = 0
        else:                           # 权重数值还要根据实际情况确定，更多类型还请自己添加
            continue
    #权重修改
    wordflagweight = np.array(wordflagweight)
    newweight = weight.copy()
    for i in range(len(weight)):                
        for j in range(len(word)):
            newweight[i][j] = weight[i][j]*wordflagweight[j]
    print("特征加权完毕！")
    new2weight =[]           
    for j in range(len(word)):
        if word[j] in word2:
            new2weight.append(list(newweight[:,j]))
    new2weight=np.array(new2weight)
    new2weight=np.transpose(new2weight)
    return new2weight

file_path=r'D:/tdb/C题全部数据/附件4.xlsx'
data = xlrd.open_workbook(file_path) #获取数据
table= data.sheet_by_name('Sheet1')
ncols = table.ncols #获取总列数
nrows=table.nrows #获取总行数

corpus,word2flagdict=corpus_segment(table,nrows,4) #留言的分词
reply,r_word2flagdict=corpus_segment(table,nrows,5) #部门回复的分词
def tf_idf(word2dict,corpus):
    vectorizer=TfidfVectorizer(max_features =len(word2dict)*2//3,max_df = 0.5,smooth_idf = True)
    tfidf=vectorizer.fit_transform(corpus)
    word=vectorizer.get_feature_names()
    weight = tfidf.toarray()
    vocab=vectorizer.vocabulary_
    return word,weight,vocab

word,weight,vocab=tf_idf(word2flagdict,corpus)
#主题和留言合并，用双文本向量表示一篇文章
newweight=new_weight(word,weight,word2flagdict)

vectorizer=TfidfVectorizer(vocabulary=vocab,max_features =len(r_word2flagdict)*2//3,max_df = 0.5,smooth_idf = True)
rtfidf=vectorizer.fit_transform(reply)
rweight=rtfidf.toarray()
rword=vectorizer.get_feature_names()
rweight=new_weight(rword,rweight,word2flagdict)

#cosine_similarity()测量任意两个或多个概要之间的相似性
similar=[]
for i in range(len(newweight)):
    dist =cosine_similarity([newweight[i],rweight[i]])
    similar.append(dist[0][1])
    
font = {'family' : 'SimHei','weight' : 'bold','size'  : '16'}
plt.rc('font', **font)
plt.rc('axes',unicode_minus=False)
plt.figure(figsize=(9,6))
plt.title("回复与留言的余弦相似度") 
plt.plot(range(len(reply)), similar, marker='o')
plt.show()

import dateutil.parser
data=pd.read_excel(r'D:/tdb/C题全部数据/附件4.xlsx')
sj=data['留言时间']
sj2=data['答复时间']
sj_cha=[]
for i in range(len(sj)):
    a=dateutil.parser.parse(str(sj2[i]))
    b=dateutil.parser.parse(str(sj[i]))
    sj_cha.append((a-b).days)
plt.figure(figsize=(9,6))
plt.title("回复的时间差") 
plt.plot(range(len(sj_cha)), sj_cha, marker='o')
plt.show()