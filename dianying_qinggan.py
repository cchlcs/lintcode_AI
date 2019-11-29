#!/usr/bin/env python
# coding: utf-8

# #导入数据集

# In[2]:


# pandas是用来导入、整理、清洗表格数据的专用工具，类似excel，但功能更加强大，导入的时候给pandas起个小名叫pd
import pandas as pd


# In[3]:


# 用pandas的read_csv函数读取训练数据及测试数据，数据文件是.tsv格式的，也就是说数据用制表符\t分隔，类似于.csv文件的数据用逗号分隔
data_train = pd.read_csv('D:\\CHENGXU\\lintcode\\dianying\\labeledTrainData.tsv',sep='\t')
data_test = pd.read_csv('D:\\CHENGXU\\lintcode\\dianying\\testData.tsv',sep='\t')


# In[4]:


# 看训练集数据前5行，Phrase列为电影评论文本，Sentiment为情感标签
data_train.head()


# In[5]:


# 共有25000行训练数据，每行数据都有句子ID、文本内容、情感标签三列
data_train.shape


# In[6]:


# 查看测试集数据前5行，Phrase列就是需要我们自己构建模型预测情感标签的文本
data_test.head()


# In[7]:


# 共有5000行测试集数据，每个数据都有句子ID、文本内容两列
data_test.shape


# 构建语料库
# 需要对文本进行一些处理，将原始文本中的每一个词变成计算机看得懂的向量，这一过程叫做文本的特征工程，非常重要。
# 有很多将词变成向量的方法，比如下面将要介绍的词袋模型、TF-IDF模型，以及word2vec模型。
# 不管采用什么模型，我们都需要先把训练集和测试集中所有文本内容组合在一起，构建一个语料库。

# In[8]:


# 提取训练集中的文本内容 
train_sentences = data_train['review']

# 提取测试集中的文本内容
test_sentences = data_test['review']

# 通过pandas的concat函数将训练集和测试集的文本内容合并到一起
sentences = pd.concat([train_sentences,test_sentences])


# In[9]:


# 合并到一起的语料库共有30000行数据
sentences.shape


# In[10]:


# 提取训练集中的情感标签，一共是25000个标签
label = data_train['sentiment']


# In[11]:


label.shape


# 使用词袋模型进行文本特征工程
# 词袋模型

# In[12]:


# 用sklearn库中的CountVectorizer构建词袋模型
# analyzer='word'指的是以词为单位进行分析，对于拉丁语系语言，有时需要以字母'character'为单位进行分析
# ngram指分析相邻的几个词，避免原始的词袋模型中词序丢失的问题
# max_features指最终的词袋矩阵里面包含语料库中出现次数最多的多少个词

from sklearn.feature_extraction.text import CountVectorizer
co = CountVectorizer(
    analyzer='word',
    ngram_range=(1,4),
    max_features=150000
)


# In[13]:


# 使用语料库，构建词袋模型

co.fit(sentences)


# In[14]:


# 将训练集随机拆分为新的训练集和验证集，默认3:1,然后进行词频统计
# 新的训练集和验证集都来自于最初的训练集，都是有标签的。

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_sentences,label,random_state=1234)


# In[15]:


# 随便看训练集中的一个数据
x_train[1]


# In[16]:


# 用上面构建的词袋模型，把训练集和验证集中的每一个词都进行特征工程，变成向量

x_train = co.transform(x_train)
x_test = co.transform(x_test)


# In[17]:


# 随便看训练集中的一个数据，它是150000列的稀疏矩阵

x_train[1]


# 构建分类器算法，对词袋模型处理后的文本进行机器学习和数据挖掘
# 逻辑回归分类器

# In[18]:


# 忽略下面代码执行过程中的版本警告等无用提示

import warnings 
warnings.filterwarnings('ignore')


# In[19]:


from sklearn.linear_model import LogisticRegression
lg1 = LogisticRegression()
lg1.fit(x_train,y_train)
print('词袋方法进行文本特征工程，使用sklearn默认的逻辑回归分类器，验证集上的预测准确率:',lg1.score(x_test,y_test))


# 多项式朴素贝叶斯分类器

# In[20]:


#引用朴素贝叶斯进行分类训练和预测
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
print('词袋方法进行文本特征工程，使用sklearn默认的多项式朴素贝叶斯分类器，验证集上的预测准确率:',classifier.score(x_test,y_test))


# 多项式朴素贝叶斯分类器，训练速度很快，但准确率较低。

# 使用TF-IDF模型进行文本特征工程

# TF值衡量了一个词出现的次数。
# IDF值衡量了这个词是不是有用。如果是the、an、a等烂大街的词，IDF值就会很低。
# 两个值的乘积TF_IDF反映了一个词的出现带来的特异性信息。

# In[21]:


# 用sklearn库中的TfidfVectorizer构建TF-IDF模型
# analyzer='word'指的是以词为单位进行分析，对于拉丁语系语言，有时需要以字母'character'为单位进行分析
# ngram指分析相邻的几个词，避免原始的词袋模型中词序丢失的问题
# max_features指最终的词袋矩阵里面包含语料库中出现次数最多的多少个词
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,4),
    max_features=150000
)


# In[22]:


tf.fit(sentences)


# 类似上面的操作，拆分原始训练集为训练集和验证集，用TF-IDF模型对每一个词都进行特征工程，变成向量

# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_sentences,label,random_state=1234)


# In[24]:


x_train = tf.transform(x_train)
x_test = tf.transform(x_test)


# In[25]:


x_train[1]


# 构建分类器算法，对TF-IDF模型处理后的文本进行机器学习和数据挖掘
# 
# 
# 朴素贝叶斯分类器

# In[26]:


#引用朴素贝叶斯进行分类训练和预测
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
print('TF-IDF方法进行文本特征工程，使用sklearn默认的多项式朴素贝叶斯分类器，验证集上的预测准确率:',classifier.score(x_test,y_test))


# 
# 逻辑回归分类器

# In[27]:


# sklearn默认的逻辑回归模型
lg1 = LogisticRegression()
lg1.fit(x_train,y_train)
print('TF-IDF方法进行文本特征工程，使用sklearn默认的逻辑回归模型，验证集上的预测准确率:',lg1.score(x_test,y_test))


# In[28]:


# C：正则化系数，C越小，正则化效果越强
# dual：求解原问题的对偶问题
lg2 = LogisticRegression(C=3, dual=True)
lg2.fit(x_train,y_train)
print('TF-IDF方法进行文本特征工程，使用增加了两个参数的逻辑回归模型，验证集上的预测准确率:',lg2.score(x_test,y_test))


# 对比两个预测准确率可以看出，在逻辑回归中增加C和dual这两个参数可以提高验证集上的预测准确率，但如果每次都手动修改就太麻烦了。我们可以用sklearn提供的强大的网格搜索功能进行超参数的批量试验。
# 搜索空间：C从1到9。对每一个C，都分别尝试dual为True和False的两种参数。
# 最后从所有参数中挑出能够使模型在验证集上预测准确率最高的。

# In[29]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C':range(1,10),
             'dual':[True,False]
              }
lgGS = LogisticRegression()
grid = GridSearchCV(lgGS, param_grid=param_grid,cv=3,n_jobs=-1)
grid.fit(x_train,y_train)


# In[30]:


grid.best_params_


# In[31]:


lg_final = grid.best_estimator_


# In[32]:


print('经过网格搜索，找到最优超参数组合对应的逻辑回归模型，在验证集上的预测准确率:',lg_final.score(x_test,y_test))


# 对测试集的数据进行预测，提交lintcode_AI竞赛最终结果

# In[33]:


# 查看测试集数据前5行，review列就是需要我们自己构建模型预测情感标签的文本
data_test.head()


# In[34]:


# 使用TF-IDF对测试集中的文本进行特征工程
test_X = tf.transform(data_test['review'])


# In[35]:


# 对测试集中的文本，使用lg_final逻辑回归分类器进行预测
predictions = lg_final.predict(test_X)


# In[36]:


predictions


# In[37]:


predictions.shape


# In[38]:


# 将预测结果加在测试集中

data_test.loc[:,'sentiment'] = predictions


# In[39]:


data_test.head()


# In[40]:


# 按lintcode官网上的要求整理成这样的格式

final_data = data_test.loc[:,['id','sentiment']]


# In[41]:


final_data.head()


# In[42]:


# 保存为.csv文件，即为最终结果

final_data.to_csv('D:\\CHENGXU\\lintcode\\dianying\\final_data1.csv',index=None)


# In[ ]:




