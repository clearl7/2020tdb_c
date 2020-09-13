# 2020-c-
题目类型是NLP自然语言处理；主要对群众问政留言记录，及相关部门对部分群众留言的答复意见进行文本挖掘。

一. 多标签留言分类
  1.1 对数据进行分析
    1.题目的目标是建立留言内容的一级标签分类模型。
    2.观察数据规模，未发现有空数据、冗余数据。
    3.附件一提供留言分类三级标签，附件二的数据主要分为主题和留言内容；可发现每一条留言的主题对分类的贡献度很大，而且能反映主题的关键词与附件一的第三类
    标签的关键词很相似，因此附件一的第三类标签可以利用。
    4.数据中涉及到多少种一级分类
    
  1.2 数据预处理
    1.对每一类随机选取2/3篇留言作为训练集，1/3作为测试集。
    2.去停用词，并进行中文分词
    3.建立词性索引词典，获取附件一第三类标签的词袋（作用：对关键词进行加权）。
    4.基于改进的TF-IDF的特征处理，特征加权（针对主题，关键词）
    
  1.3 朴素贝叶斯模型
    使用朴素贝叶斯训练一个多分类模型，输入使用词频向量（经过tf-idf特征提取），使用测试集测试，并计算F-score。
    
二. 热点挖掘
    2.1 问题目标
       挖掘在某一时间段内反应特定地点和特定人群的热点问题。此外，还需要建立一个热度评价指标，这可能与：每个事件被反映的次数、点赞数、反对数等有关。
    
    2.2 数据预处理
        1.去停用词，进行中文分词
        2.文本信息向量化，基于TF-IDF的特征提取并加权（针对主题，关键词）
        3.文本相似度计算
        
    2.3 基于余弦相似度的谱聚类分析模型
        1.得到文本相似度矩阵，对其使用谱聚类算法，多次迭代得到轮廓系数最高的蔟。
        
    2.4 找出点赞数与反对数最高的留言
        1.利用余弦相似度找出与点赞数与反对数最高的同类型留言。
    
    2.5 热度评价指标
        设每个事件被反映的次数m、点赞数z、反对数r,自定义建立这个指标的评价模型
        
 三. 答复意见的评价
    1.相似度：文本相似度
    2.时效性：意见与回复时间差
    3.完整性：正则匹配特定格式
    将指标量化分析综合评定
