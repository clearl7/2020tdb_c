# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:47:13 2020

@author: acer
"""

import xlwt,xlrd
import os
import codecs
import jieba.posseg as pseg
import re
import heapq
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def Stopword():
    """导入停用词"""
    os.chdir('D:/')
    stopword=[w.strip() for w in codecs.open("stopwords.txt",'r','utf-8').readlines()]
    return stopword

def corpus_segment(table,n,v):
    """
    中文分词并去停用词
    return:分词后的文本，字典
    """
    
    stopword=Stopword()
    out_get=open(r"D:/hotspot.txt",'w',encoding="utf-8")
    evaluate_get=open(r"D:/evaluate.txt",'w')
    cixingofword = []  # 储存分词后的词语对应的词性
    wordtocixing = []  # 储存分词后的词语
    cutcorpus=[0]*(n-1)
    for i in range(n-1):
        dianzan=str(int(table.cell(i+1,5).value)+int(table.cell(i+1,6).value))
        evaluate_get.write(dianzan+'\n')
        content=table.cell(i+1,v).value
        content1 = re.sub(r'[^\u4e00-\u9fa5]', "",content)#只保留中文
        content1 = pseg.cut(content1)
        cutcorpus[i] = ""
        for words,cixing in content1:
            if words.strip() not in stopword and len(words.strip())>1:
                cutcorpus[i] =cutcorpus[i]+words.strip()+' '
                cixingofword.append(cixing)
                wordtocixing.append(words)
        out_get.write(cutcorpus[i]+"\n")
    out_get.close()
    evaluate_get.close()
    print("文本分词结束！")
    # 自己造一个{“词语”:“词性”}的字典，方便后续使用词性
    word2flagdict = {wordtocixing[i]:cixingofword[i] for i in range(len(wordtocixing))}
    return cutcorpus,word2flagdict

def tf_idf(word2dict,corpus,l):
    """
    计算TFIDF权重
    return：词袋，TFIDF权重矩阵
    """
    vectorizer=TfidfVectorizer(max_features =len(word2dict)*l//5,max_df = 0.5,smooth_idf = True)
    tfidf=vectorizer.fit_transform(corpus)
    word=vectorizer.get_feature_names()
    weight = tfidf.toarray()
    return word,weight

def new_weight(word,weight,word_dict):
    """
    权重的进一步改进，保留名词，动词，名动词，地方词
    word：词袋
    word_dict:字典
    weight：TFIDF权重矩阵
    return:权重修改后的矩阵
    """
    wordflagweight = [1 for i in range(len(word))]   #这个是词性系数，需要调整系数来看效果
    word2=[]
    for i in range(len(word)):
        if(word_dict[word[i]]=="n"):  # 名词重要一点，我们就给它1.25
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
        elif(word_dict[word[i]]=="m"):  # 这种量词什么的直接去掉
            wordflagweight[i] = 0
        else:                           
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

def spectral_clustering(X_dist,newweight):
    from numpy import linalg as LA
    from sklearn.preprocessing import normalize
    """
    谱聚类
    :X_dist:相似度矩阵
    :return: 聚类结果，两个优蔟
    """
    font = {'family' : 'SimHei','weight' : 'bold','size'  : '16'}
    print("谱聚类开始")
    Dn = np.diag(np.power(np.sum(X_dist, axis=1), -0.5)) #度矩阵
    L = np.eye(len(X_dist)) - np.dot(np.dot(Dn, X_dist), Dn) # 拉普拉斯矩阵：L=Dn*(D-W)*Dn=I-Dn*W*Dn
    eigvals, eigvecs = LA.eig(L)   #计算得到特征值和特征向量
    print("开始调参")
    scores=[]
    for k in range(8,60):
        indices = np.argsort(eigvals)[:k]   # 前k小的特征值对应的索引，argsort函数
        k_smallest_eigenvectors = normalize(eigvecs[:, indices].astype(float)) # 取出前k个最小的特征值对应的特征向量，并进行正则化
        km=cluster.MiniBatchKMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=300,)
        km.fit(k_smallest_eigenvectors)
        score=metrics.silhouette_samples(k_smallest_eigenvectors, km.labels_)
        sil_list=[]
        for i in range(k):
            sum_sil=0
            t=0
            for j in range(len(score)):
                if km.labels_[j]==i:
                    sum_sil+=score[j]
                    t+=1
            sil_list.append(sum_sil/t)
        Optimal_dist=max(sil_list) #提取局部最优来代替整体聚类的轮郭系数
        scores.append(Optimal_dist) #不同蔟中心的聚类的轮郭系数
    plt.rc('font', **font)
    plt.rc('axes',unicode_minus=False)
    plt.figure(figsize=(9,6))
    plt.title("寻找最优的簇中心个数") 
    plt.plot(range(8,60), scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('局部最优的silhouette_scores')
    plt.show()
    print("迭代完毕，选出最佳簇中心个数")
    max_score=max(scores)
    cu=scores.index(max_score)+8
    print("最佳蔟中心的个数：",cu)
    indices = np.argsort(eigvals)[:cu]   # 前k小的特征值对应的索引，argsort函数
    k_smallest_eigenvectors = normalize(eigvecs[:, indices].astype(float)) # 取出前k个最小的特征值对应的特征向量，并进行正则化
    km.n_clusters=cu #选定最优蔟中心个数
    y_label=km.fit_predict(k_smallest_eigenvectors) #聚类
    X_reduction=PCA(2).fit_transform(k_smallest_eigenvectors) #降维
    #聚类的可视化
    plt.rc('font', **font)
    plt.rc('axes',unicode_minus=False)
    plt.figure(figsize=(6,4))
    plt.title("文本的分布") 
    plt.scatter(X_reduction[:,0],X_reduction[:,1],c=y_label,edgecolor='none')
    plt.show()
    print("聚类的平均轮廓系数为：",max_score)
    score=metrics.silhouette_samples(k_smallest_eigenvectors, y_label) #每个文本的轮廓系数
    sil_list=[] #每个蔟的轮廓系数平均值
    for i in range(cu):
        sum_sil=0
        t=0
        for j in range(len(score)):
            if y_label[j]==i:
                sum_sil+=score[j]
                t+=1
        sil_list.append(sum_sil/t)
    #轮廓系数可视化
    plt.rc('font', **font)
    plt.rc('axes',unicode_minus=False)
    plt.figure(figsize=(9,6))
    plt.plot(range(cu), sil_list, marker='o')
    plt.xlabel('每个蔟中心的轮廓系数')
    plt.show()
    max_tuan = map(sil_list.index, heapq.nlargest(2, sil_list)) #求最大的两个索引
    print("聚类结束")
    return y_label,list(max_tuan)

def findtext(y,max_tuan,zanlist):
    """
    把五类问题写进表格
    y：聚类的结果
    max_tuan:两个最优蔟
    zanlist：点赞数最多的三类话题
    """
    data1 = xlrd.open_workbook(r'D:/tdb/C题全部数据/附件3.xlsx') #获取数据
    table1= data1.sheet_by_name('Sheet1')
    workbook=xlwt.Workbook(encoding='utf-8')
    booksheet=workbook.add_sheet('Sheet1', cell_overwrite_ok=True)
    
    style = xlwt.XFStyle() # 初始化样式
    font = xlwt.Font() # 为样式创建字体
    font.name = '宋体' 
    font.height=20*11
    style.font = font # 设定样式
    style1 = xlwt.XFStyle() #日期的样式
    style1.num_format_str = 'yyyy-mm-dd hh:mm:ss'
    booksheet.col(3).width = 25 * 256
    booksheet.col(4).width = 19 * 256
    booksheet.col(5).width = 15 * 256
    
    count=0
    for i in range(len(max_tuan)):
        for k in range(len(y)):
            if y[k]==max_tuan[i]:
                count+=1
                contents=table1.row_values(k+1)#获取第w行的内容
                booksheet.write(count,0,i+1,style)
                for j,content in enumerate(contents):
                    if j==3:
                        booksheet.write(count,j+1,content,style1) #输入日期
                    else:
                        booksheet.write(count,j+1,content,style)
    for l in range(len(zanlist)):
        for w in zanlist[l]:
            count+=1
            contents=table1.row_values(w+1)#获取第w行的内容
            booksheet.write(count,0,l+3,style)
            for j,content in enumerate(contents):
                if j==3:
                    booksheet.write(count,j+1,content,style1) #输入日期
                else:
                    booksheet.write(count,j+1,content,style)
    top=['问题ID	','留言编号','留言用户','留言主题','留言时间','留言详情','反对数','点赞数']
    for t in range(len(top)):
        booksheet.write(0,t,top[t],style)
    workbook.save(r'D:/tdb/C题全部数据/热点问题留言详细表.xlsx')  # 保存工作簿
    print("写入完毕！")

file_path=r'D:/tdb/C题全部数据/附件3.xlsx'
data = xlrd.open_workbook(file_path) #获取数据
table= data.sheet_by_name('Sheet1')
ncols = table.ncols #获取总列数
nrows=table.nrows #获取总行数

print('分词开始')
T_corpus,T_word2flagdict=corpus_segment(table,nrows,2) #主题的分词
t_corpus,t_word2flagdict=corpus_segment(table,nrows,4) #留言的分词
print('计算TFIDF权重')
T_word,T_weight=tf_idf(T_word2flagdict,T_corpus,4)
t_word,t_weight=tf_idf(t_word2flagdict,t_corpus,3)
#主题和留言合并，用双文本向量表示一篇文章
weight=np.concatenate((T_weight*0.7,t_weight*0.3),axis=1) #主题重要一点则权重*0.7，留言的权重*0.3
word=T_word+t_word #主题和留言的词袋合并
word2flagdict=dict(T_word2flagdict,**t_word2flagdict) #主题和留言的字典合并
newweight=new_weight(word,weight,word2flagdict) #新的权重矩阵
#测量多个文本之间的相似性
dist =cosine_similarity(newweight)
y,max_tuan=spectral_clustering(dist,newweight) #聚类

#提取点赞数最高的前三类留言
evaluate=[int(w.strip()) for w in codecs.open("evaluate.txt",'r').readlines()]
max_zan = map(evaluate.index, heapq.nlargest(3, evaluate)) #求最大的三个索引
max_zan=list(max_zan)
#找出与点赞数最高的前三类留言的相似文章
zanlist=[]
for i in max_zan:
    z=dist[i][:]
    zanlist.append([j for j in range(len(z)) if z[j]>0.361])
print("模型求解完毕，正在把五类热门话题写入文档")
findtext(y,max_tuan,zanlist)

def heat_evaluation(t):
    data = xlrd.open_workbook(r'D:/tdb/C题全部数据/热点问题留言详细表.xlsx') #获取数据
    table= data.sheet_by_name('Sheet1')
    nrows=table.nrows
    click_number=0 #统计点赞数和反对数
    number=0 #统计留言数目
    for n in range(1,nrows):
        if t==table.cell(n,0).value:
            click_number+=table.cell(n,6).value+table.cell(n,7).value
            number+=1
    return number,click_number

#计算热度指数
print("计算热度指数")
hotpot_evaluate_list=[[178],[238],[189],[222],[8]]
for i in range(1,5+1):
    a,b=heat_evaluation(i)
    hotpot_evaluate_list[i-1].append(a)
    hotpot_evaluate_list[i-1].append(b)
value=[5,3,2] #留言篇数最重要所以给的权重为5，其次是点赞的权重为3，报道时间长度的权重为2
hotpot_evaluate=np.array(hotpot_evaluate_list)
for i in range(len(hotpot_evaluate[0])):
    target=hotpot_evaluate[:,i]
    imax=max(target)
    imin=min(target)
    hotpot_evaluate[:,i]=(hotpot_evaluate[:,i]-imin)/(imax-imin)
    hotpot_evaluate[:,i]=hotpot_evaluate[:,i]*value[i]
for i in range(len(hotpot_evaluate)):
    print('问题%d的评价指数为：'%i,sum(hotpot_evaluate[i]))
