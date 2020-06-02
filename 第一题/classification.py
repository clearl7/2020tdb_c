# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:44:17 2020

@author: acer
"""

import xlrd
import os
import codecs
import jieba
import jieba.posseg as pseg
import re
import math
import numpy as np
from collections import Counter #统计词频
import pandas as pd
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法包
from sklearn import metrics # 计算分类精度：
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def Stopword():
    """
    加载停用词
    """
    os.chdir('D:/')
    stopword=[w.strip() for w in codecs.open("stopwords.txt",'r','utf-8').readlines()]
    return stopword

def count_labelcontent(table,nrows,labels):
    """
    统计各类标签的文本总数
    """
    amount=[]
    i=0
    num=0
    for j in range(1,nrows):
        if table.cell(j,5).value==labels[i]:
            num+=1
        else:
            amount.append(num)
            i+=1
            num=1
    amount.append(num)
    return amount

def corpus_segment(table,label,num,v):
    """
    中文文本分词并去停用词
    label：7类标签
    return：训练集和测试集的分词后的文本和字典
    """
    stopword=Stopword()
    out_train=open(r"D:/train_corpus.txt",'w',encoding="utf-8")
    out_test=open(r"D:/test_corpus.txt",'w',encoding="utf-8")
    out_label_train=open(r"D:/Label_train.txt",'w',encoding="utf-8")
    out_label_test=open(r"D:/Label_test.txt",'w',encoding="utf-8")
    content_train=[] #训练集文本的分词
    cixing_train=[] # 储存分词后的词语对应的词性
    wordtocixing_train=[] # 储存分词后的词语
    content_test=[] #测试集文本的分词
    cixing_test=[] # 储存分词后的词语对应的词性
    wordtocixing_test=[] # 储存分词后的词语
    j=1
    for k in range(len(num)):
        for i in range(j,j+num[k]):
            content=table.cell(i,v).value
            content1 = re.sub(r'[^\u4e00-\u9fa5]', "",content)#只保留中文
            content1 = pseg.cut(content1)
            cutcorpus= ""
            if i<=j+num[k]*2//3:   #取标签k总文本数的2/3为训练集
                for words,cixing in content1:
                    if words.strip() not in stopword and len(words.strip())>1: #去停用词
                       cutcorpus+=words.strip()+' '
                       cixing_train.append(cixing)
                       wordtocixing_train.append(words)
                content_train.append(cutcorpus)
                out_train.write(cutcorpus+"\n")
                out_label_train.write(label[k]+'\n')
            else: #剩下的1/3作为测试集
                for words,cixing in content1:
                    if words.strip() not in stopword and len(words.strip())>1:
                       cutcorpus+=words.strip()+' '
                       cixing_test.append(cixing)
                       wordtocixing_test.append(words)
                content_test.append(cutcorpus)
                out_test.write(cutcorpus+"\n")
                out_label_test.write(label[k]+'\n')
        j+=num[k]
    out_train.close()
    out_test.close()
    out_label_train.close()
    out_label_test.close()
    # 自己造一个{“词语”:“词性”}的字典，方便后续使用词性
    word2flagdict_train = {wordtocixing_train[i]:cixing_train[i] for i in range(len(wordtocixing_train))}
    word2flagdict_test = {wordtocixing_test[i]:cixing_test[i] for i in range(len(wordtocixing_test))}
    print("文本分词结束！")
    return content_train,content_test,word2flagdict_train,word2flagdict_test

def corpus2_segment(table,label,n):
    """
    把附件一的三级分类进行分词
    """
    stopword=Stopword()
    three_class=[]
    
    for l in label:
        label_content=""
        for i in range(1,n):
            if table.cell(i,0).value==l:
                label_content+=table.cell(i,1).value+' '+table.cell(i,2).value+' '
        label_content= re.sub(r'[^\u4e00-\u9fa5]', "",label_content)#只保留中文
        label_content = jieba.lcut(label_content)
        label_content=set(label_content)
        label_content = ' '.join(w for w in label_content if w not in stopword)#去停用词
        three_class.append((label_content))
    return three_class

def quchong(tclass):
    """
    对附件一的每一类的分词后文本去重，筛选出更有代表意义的关键性
    return：每一类独有的关键词集合
    """
    t2class=[]
    s=set()
    t3class=[]
    for t in tclass:
        l=set(t.split())
        t2class.append(l)
    for i in range(len(t2class)):
        for j in range(i+1,len(t2class)):
            s=s|(t2class[i]&t2class[j])
    for k in range(len(t2class)):
        t3class.append(t2class[k]-s)
    return t3class

def Word_frequency(ct_train,ct_test):
    """
    统计每一个文本的词频
    ct_train：训练集
    ct_test：测试集
    return：vsm向量
    """
    print('计算文本单词的词频')
    trainlist=[]
    testlist=[]
    for w in ct_train:
        trainlist.append(w.split())
    for w2 in ct_test:
        testlist.append(w2.split())
    train_count=[]
    test_count=[]
    for i in range(len(trainlist)):
        count=Counter(trainlist[i])
        train_count.append(count)
    for j in range(len(testlist)):
        count2=Counter(testlist[j])
        test_count.append(count2)
    return train_count,test_count

def tf(word,count):
    """
    计算特征的tf值
    """
    return count[word]/sum(count.values())

def n_containing(word, count_list,lcontent,clabel):
    """
    统计在第i类包含该词的文本数，以及类外包含该次的文本数
    """
    m=0;j=0;k=0;i=0
    for l in range(len(lcontent)):
        if clabel==lcontent[l]:
            if word in count_list[l]:
                m+=1
            else:
                j+=1
        else:
            if word in count_list[l]:
                k+=1
            else:
                i+=1
    return m,j,k,i

def idf(word,clabel,clab,counter):
    """
    计算特征的idf值
    """
    m,j,k,i=n_containing(word,counter,clabel,clab)
    d=m/(m+j)
    b=k/(k+i)
    N=len(counter)
    return math.log(d/(d+b)*N)

def tfidf(word,count,clabel,clab,counter):
    """
    计算tf*idf值
    """
    t_class=quchong(three_class)
    if word in t_class[labels.index(clab.strip('\n'))]:
        return 1.3*tf(word,count)*idf(word,clabel,clab,counter)
    else:
        return tf(word,count)*idf(word,clabel,clab,counter)

def w_value(count_list,clabel,n):
    """
    特征加权并选择
    return：带tfidf权重的vsm文本向量模型
    """
    wtfidf_list=[]
    lab=0
    for cl in count_list:
        wtfidf={}
        for key in cl:
            wtfidf[key]=tfidf(key,cl,clabel,clabel[lab],count_list) #计算每个文本中各词的TFIDF值
        wtfidf_list.append(wtfidf)
        lab+=1
    wtfidf_vsm=[]
    #选取TFIDF值top30的词作为关键词
    for cc in wtfidf_list:
        z1=zip(cc.values(),cc.keys())
        z1=list(sorted(z1))
        z1.reverse()
        z1=z1[:n]
        wtfidf_vsm.append(z1)
    return wtfidf_vsm


def vacablary(wtfidf_vsm):
    """
    返回训练集词袋
    """
    vacab=set()
    for w in wtfidf_vsm:
        l=[]
        for k in w:
            l.append(k[1])
        vacab=vacab|set(l)
    return list(vacab)
#
def juzhen(vocab,wtfidf_vsm):
    """
    返回带TFIDF权值的矩阵
    """
    returnVec_all=[]
    for w in wtfidf_vsm:
        returnVec=[0]*len(vocab)
        for j in w:
            returnVec[vocab.index(j[1])]=j[0]
        returnVec_all.append(returnVec)
    return  returnVec_all

def testWord2Vec(wtfidf2_vsm,vocab):
    """
    返回测试集的tfidf矩阵
    """
    testVec_all=[]
    for w in wtfidf2_vsm:
        testVec=[0]*len(vocab)
        for j in w:
            if j[1] in vocab:
                testVec[vocab.index(j[1])]=j[0]
        testVec_all.append(testVec)
    return testVec_all

def readfile(path):
    fp = open(path,"r",encoding='utf-8',errors='ignore')
    content = fp.readlines()
    fp.close()
    return content

def metrics_result(actual, predict,label):
    """
    计算精度
    """
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict,average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict,average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict,average='weighted')))
    # 计算各个类别的准确率，召回率，与F1-score
    p_class, r_class, f_class, support_micro = metrics.precision_recall_fscore_support(actual, predict)
    font = {'family' : 'SimHei','weight' : 'bold','size'  : '16'}
    plt.rc('font', **font)
    plt.rc('axes',unicode_minus=False)
    plt.figure(figsize=(14,6))
    plt.title("每个类的f-score")  # 指定标题，并设置标题字体大小
    plt.plot(labels,f_class,color="red",linewidth=5)
    plt.show()
    return f_class

def idf2(word,count_list):
    N=len(count_list)
    t=0
    for i in count_list:
        if word in i:
            t+=1
    return math.log(N/t)

def tfidf2(word,cl,count_list):
    """
    计算tf*idf值
    """
    return tf(word,cl)*idf2(word,count_list)
def w2_value(count_list,n):
    """
    特征加权并选择
    return：带tfidf权重的vsm文本向量模型
    """
    wtfidf_list=[]
    for cl in count_list:
        wtfidf={}
        for key in cl:
            wtfidf[key]=tfidf2(key,cl,count_list) #计算每个文本中各词的TFIDF值
        wtfidf_list.append(wtfidf)
    wtfidf_vsm=[]
    #选取TFIDF值top30的词作为关键词
    for cc in wtfidf_list:
        z1=zip(cc.values(),cc.keys())
        z1=list(sorted(z1))
        z1.reverse()
        z1=z1[:n]
        wtfidf_vsm.append(z1)
    return wtfidf_vsm

file_path=r'D:/tdb/C题全部数据/附件2.xlsx'
data = xlrd.open_workbook(file_path) #获取数据
table= data.sheet_by_name('Sheet1')
nrows=table.nrows #获取总行数
file_path2=r'D:/tdb/C题全部数据/附件1.xlsx'
data2 = xlrd.open_workbook(file_path2) #获取数据
table2= data2.sheet_by_name('Sheet1')
nrows2=table2.nrows #获取总行数

labels=['城乡建设','环境保护','交通运输','教育文体','劳动和社会保障','商贸旅游','卫生计生']
print("开始对训练集，测试集进行分词")
labelcontent=count_labelcontent(table,nrows,labels) #统计各类标签的文本总数
#主题和留言分开，用双文本向量表示一篇文章
theme_train,theme_test,theme2dict_train,theme2dict_test=corpus_segment(table,labels,labelcontent,2)
liuyan_train,liuyan_test,liuyan2dict_train,liuyan2dict_test=corpus_segment(table,labels,labelcontent,4)
three_class=corpus2_segment(table2,labels,nrows2) #获得附件一的三级分类的关键词

train_label=readfile(r'D:\Label_train.txt') #每一条留言对应的分类
test_label=readfile(r'D:\Label_test.txt')   #每一条留言对应的分类

print("开始计算文本词语的词频")
theme_train_count,theme_test_count=Word_frequency(theme_train,theme_test) #计算文本单词的词频
liuyan_train_count,liuyan_test_count=Word_frequency(liuyan_train,liuyan_test) #计算文本单词的词频

#特征提取，并计算改进的TFIDF值
print('开始训练训练集')
#构建主题的TFIDF权重矩阵
theme_wtfidf_vsm=w_value(theme_train_count,train_label,20) #特征提取top20的关键词
theme_vacab=vacablary(theme_wtfidf_vsm) #获得词袋
theme_words2Vec=juzhen(theme_vacab,theme_wtfidf_vsm)
#构建留言的TFIDF权重矩阵
liuyan_wtfidf_vsm=w_value(liuyan_train_count,train_label,30) #特征提取top30的关键词
liuyan_vacab=vacablary(liuyan_wtfidf_vsm) #获得词袋
liuyan_words2Vec=juzhen(liuyan_vacab,liuyan_wtfidf_vsm)
#主题与留言的矩阵合并
words2Vec=np.concatenate((np.array(theme_words2Vec)*0.7,np.array(liuyan_words2Vec)*0.3),axis=1) #主题重要一点则权重*0.7，留言的权重*0.3
vacab=theme_vacab+liuyan_vacab
px=pd.DataFrame(words2Vec,columns=vacab)
py=pd.DataFrame(train_label)

print('开始训练测试集')
#构建主题的TFIDF权重矩阵
theme_wtfidf2_vsm=w2_value(theme_test_count,20) #特征提取top20的关键词
theme_test2Vec=testWord2Vec(theme_wtfidf2_vsm,theme_vacab)
#构建留言的TFIDF权重矩阵
liuyan_wtfidf2_vsm=w2_value(liuyan_test_count,30) #特征提取top30的关键词
liuyan_test2Vec=testWord2Vec(liuyan_wtfidf2_vsm,liuyan_vacab)
#主题与留言的矩阵合并
test2Vec=np.concatenate((np.array(theme_test2Vec)*0.7,np.array(liuyan_test2Vec)*0.3),axis=1) #主题重要一点则权重*0.7，留言的权重*0.3
pt=pd.DataFrame(test2Vec,columns=vacab)

# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
print("开始分类")
clf = MultinomialNB(alpha=0.0001).fit(px,py)
# 预测分类结果
predicted = clf.predict(pt)
#分类结果的可视化
y=[0]*len(test_label)
for i in range(len(labels)):
    for j in range(len(test_label)):
        if test_label[j].strip()==labels[i]:
            y[j]=i
X_reduction=PCA(2).fit_transform(pt) #降维
plt.figure(figsize=(4,4))
plt.scatter(X_reduction[:,0],X_reduction[:,1],c=y,edgecolor='none')
plt.show()

for flabel,expct_cate in zip(test_label,predicted):
    if flabel != expct_cate:
        print(": 实际类别:",flabel," -->预测类别:",expct_cate)
print("预测完毕!!!")
f_class=metrics_result(test_label, predicted,labels)