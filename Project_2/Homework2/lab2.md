# <center>智能推荐系统第二次编程作业

## 						   ——Content-based Recommendation















### 学号：10185102142

### 姓名：李泽浩

### 指导老师：张伟

### 项目名称：Content-based Recommendation

### 时间：2021年5月5日



<div style="page-break-after:always"></div>



## <center>目录<center>

### [1.基于内容的推荐算法](###1、基于内容的推荐算法简介)

#### 1.1 算法详解

#### 1.2 物品表示

#### 1.3 特征学习

#### 1.4 优缺点

#### 1.5 总结

### 2.[代码](###2、代码详解)

#### 对训练集进行建模

### 3.[结果分析及准确率](###3、结果分析及准确率)

#### 对测试集数据进行分析

#### 准确率分析

#### 误差分析

### 4.[提交文件](###4、提交文件列表)

#### 4.1实验报告PDF

#### 4.2Python代码 source_code.ipynb

#### 4.3预测补充后的test.tsv文件

<div style="page-break-after: always;"></div>

### 1、基于内容的推荐算法简介

#### 1.1算法详解

CB是最早被使用的推荐算法，它的思想非常简单：根据用户过去喜欢的物品（本文统称为 item），为用户推荐和他过去喜欢的物品相似的物品。而关键就在于这里的物品相似性的度量，这才是算法运用过程中的核心。 CB最早主要是应用在信息检索系统当中，所以很多信息检索及信息过滤里的方法都能用于CB中。

​		举个简单的例子：在京东上购物的小伙伴们应该都知道，每当你进入任何一个物品页面的时候都会有一个“猜你喜欢”的栏目，这时候他就会根据你经常购买的物品给你推荐相似的物品。例如对我来说：我经常购买互联网类书籍，所以它就会给我推荐类似的书籍。

CB的过程一般包括以下三步：

- 物品表示（Item Representation）：为每个item抽取出一些特征（也就是item的content了）来表示此item；

- 特征学习（Profile Learning）：利用一个用户过去喜欢（及不喜欢）的item的特征数据，来学习出此用户的喜好特征（profile）
- 生成推荐列表（Recommendation Generation）：通过比较上一步得到的用户profile与候选item的特征，为此用户推荐一组相关性最大的item。

举个例子说明前面的三个步骤。随着今日头条的崛起，基于内容的文本推荐就盛行起来。在这种应用中一个item就是一篇文章。

​	第一步，我们首先要从文章内容中抽取出代表它们的属性。常用的方法就是利用出现在一篇文章中词来代表这篇文章，而每个词对应的权重往往使用信息检索中的tf-idf来计算。利用这种方法，一篇抽象的文章就可以使用具体的一个向量来表示了。

​	第二步，根据用户过去喜欢什么文章来产生刻画此用户喜好的特征向量了，最简单的方法可以把用户所有喜欢的文章对应的向量的平均值作为此用户的特征向量。比如我经常在今日头条阅读技术科技相关的文章，那么今日头条的算法可能会把我的Profile中的：“互联网”、“大数据”、“机器学习”、“数据挖掘”等关键词的权重设置的比较大。	

​	这样，当我登录头条客户端的时候，他获取到我的用户Profile后，利用CB算法将我的个人Profile与文章Item的Profile的相似度（相似度的衡量可以用余弦相似度Cosine）进行计算，然后按相似度大小取最大的前N个篇文章作为推荐结果返回给我的推荐列表中。

#### 1.2物品表示

真实应用中的item往往都会有一些可以描述它的属性。这些属性通常可以分为两种：结构化的（structured）属性与非结构化的（unstructured）属性。所谓结构化的属性就是这个属性的意义比较明确，其取值限定在某个范围；而非结构化的属性往往其意义不太明确，取值也没什么限制，不好直接使用。比如在交友网站上，item就是人，一个item会有结构化属性如身高、学历、籍贯等，也会有非结构化属性（如item自己写的交友宣言，博客内容等等）。对于结构化数据，我们自然可以拿来就用；但对于非结构化数据（如文章），我们往往要先把它转化为结构化数据后才能在模型里加以使用。真实场景中碰到最多的非结构化数据可能就是文章了（如个性化阅读中）。

#### 1.3特征学习的主要方法

假设用户u已经对一些item给出了他的喜好判断，喜欢其中的一部分item，不喜欢其中的另一部分。那么，这一步要做的就是通过用户u过去的这些喜好判断，为他产生一个模型。有了这个模型，我们就可以根据此模型来判断用户u是否会喜欢一个新的item。所以，我们要解决的是一个典型的有监督分类问题，理论上机器学习里的分类算法都可以照搬进这里。

主要方法有：

- Rocchio算法
- 最近邻方法（简称KNN）

#### 1.4优缺点

CB的优点：

- 用户之间的独立性（User Independence）：既然每个用户的profile都是依据他本身对item的喜好获得的，自然就与他人的行为无关。而CF刚好相反，CF需要利用很多其他人的数据。CB的这种用户独立性带来的一个显著好处是别人不管对item如何作弊（比如利用多个账号把某个产品的排名刷上去）都不会影响到自己。
- 好的可解释性（Transparency）：如果需要向用户解释为什么推荐了这些产品给他，你只要告诉他这些产品有某某属性，这些属性跟你的品味很匹配等等。
- 新的item可以立刻得到推荐（New Item Problem）：只要一个新item加进item库，它就马上可以被推荐，被推荐的机会和老的item是一致的。而CF对于新item就很无奈，只有当此新item被某些用户喜欢过（或打过分），它才可能被推荐给其他用户。所以，如果一个纯CF的推荐系统，新加进来的item就永远不会被推荐:( 。

CB的缺点：

- item的特征抽取一般很难（Limited Content Analysis）：如果系统中的item是文档（如个性化阅读中），那么我们现在可以比较容易地使用信息检索里的方法来“比较精确地”抽取出item的特征。但很多情况下我们很难从item中抽取出准确刻画item的特征，比如电影推荐中item是电影，社会化网络推荐中item是人，这些item属性都不好抽。其实，几乎在所有实际情况中我们抽取的item特征都仅能代表item的一些方面，不可能代表item的所有方面。这样带来的一个问题就是可能从两个item抽取出来的特征完全相同，这种情况下CB就完全无法区分这两个item了。比如如果只能从电影里抽取出演员、导演，那么两部有相同演员和导演的电影对于CB来说就完全不可区分了。
- 无法挖掘出用户的潜在兴趣（Over-specialization）：既然CB的推荐只依赖于用户过去对某些item的喜好，它产生的推荐也都会和用户过去喜欢的item相似。如果一个人以前只看与推荐有关的文章，那CB只会给他推荐更多与推荐相关的文章，它不会知道用户可能还喜欢数码。
- 无法为新用户产生推荐（New User Problem）：新用户没有喜好历史，自然无法获得他的profile，所以也就无法为他产生推荐了。当然，这个问题CF也有。

#### 1.5总结

CB应该算是第一代的个性化应用中最流行的推荐算法了。但由于它本身具有某些很难解决的缺点（如上面介绍的第1点），再加上在大多数情况下其精度都不是最好的，目前大部分的推荐系统都是以其他算法为主（如CF），而辅以CB以解决主算法在某些情况下的不精确性（如解决新item问题）。但CB的作用是不可否认的，只要具体应用中有可用的属性，那么基本都能在系统里看到CB的影子。组合CB和其他推荐算法的方法很多（我很久以后会写一篇博文详细介绍之），最常用的可能是用CB来过滤其他算法的候选集，把一些不太合适的候选（比如不要给小孩推荐偏成人的书籍）去掉。

<div style="page-break-after:always"></div>

<div style="page-break-after:always"></div>

### 2、代码详解

##### 2.1导入必要的函数库

```python
#导入必要的函数库
import numpy
import random
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import defaultdict
```

##### 2.2读取并查看训练集数据

```python
#训练集
train_news_path = r"./train/train_news.tsv"
train_history_path = r"./train/train.tsv"

train_history = pd.read_csv(train_history_path, sep='\t',header=0)

train_news = pd.read_csv(train_news_path, sep='\t',header=0)
```

```python
#测试集


```

##### 2.3新闻归类

```python
#将新闻按照Category归类，归类后存入new_dateset
n = open(train_news_path,"rt",encoding="utf-8")
header = n.readline()
header = header.strip().split('\t')#列名
print(header)

news_dataset = []

for line in n:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    news_dataset.append(d)
   
News_Category = defaultdict(set)

for d in news_dataset:
    nid, category = d['Nid'],d["Category"]
    News_Category[category].add(nid)
```

##### 2.4用户历史浏览记录

```python
u = open(train_history_path,"rt",encoding="utf-8")
header2 = u.readline()
header2 = header2.strip().split('\t')#列名
print(header2)

users_dataset = []

for line in u:
    fields2 = line.strip().split('\t')
    u = dict(zip(header2, fields2))
    users_dataset.append(u)

User_list = defaultdict(set)

for d in users_dataset:
    uid, history  = d["Uid"],d['History']
    User_list[uid].add(history)
```

##### 2.5数据分析

```python
UserPerItem = defaultdict(set)	#记录对某商品评价过的用户
ItemPerUser = defaultdict(set)	#记录某用户评价过的商品
#把训练集数据由dataset读入上述字典中
for d in dataset:
    user,item = d['user_id'], d['business_id']
    UserPerItem[item].add(user)#评价过某商品的用户列表，key值为business_id
    ItemPerUser[user].add(item)#某用户评价过的商品列表，key值为用户名user_id
```

##### 2.6相似度函数

```python
#余弦相似度
def cos(s1,s2):
    demon = 0.0
    number = len(s1.intersection(s2))#并集
    l1 = len(s1) #s1集合中元素个数
    l2 = len(s2) #s2集合中元素个数
    demon += math.sqrt(l1 * l2)
    if demon == 0:
        return 0
    return number / demon
```

##### 2.7预测函数

```python
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    user,item = d['user_id'], d['business_id']
    reviewsPerItem[item].append(d)
    reviewsPerUser[user].append(d)

def prdictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['business_id']
        if i2 == item:continue
        ratings.append(d['stars'])
        similarities.append(cos(UserPerItem[item],UserPerItem[i2]))
    if(sum(similarities)>0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean
```



<div style="page-break-after:always"></div>

### 3、结果分析及准确率

##### 3.1读取测试集数据预测并写回

```python
#读取测试集数据
filename2 = "test.csv"
f2 = open(filename2, "rt", encoding="utf-8")
header2 = f2.readline()
header2 = header2.strip().split(',')#列名
print(header2)
header2.append('pre_stars')
print(header2)

predata = []

for line in f2:
    fields = line.strip().split(',')
    d = dict(zip(header2, fields))
    d["pre_stars"] = 0
    predata.append(d)

for d in predata:
    u, i = d['user_id'], d['business_id']
    s = round(prdictRating(u,i))
    d['pre_stars'] = s

df = pd.DataFrame(predata)
df.to_csv(filename2)
```

##### 3.2对训练集进行预测并查看准确率

```python
s = len(dataset)
count = 0
for d in dataset:
    user,item,star = d['user_id'], d['business_id'], d['stars']
    star = float(star)
    p = round(prdictRating(user,item))
    if p - star <= 0.5:
        count += 1
print(count/s)
```

结果如下：

<img src="/Users/lee/Study/大三-下/智能推荐系统/Project_1/lab1.assets/截屏2021-03-31 下午10.47.53.png" alt="截屏2021-03-31 下午10.47.53" style="zoom:67%;" />

##### 3.3计算误差

```python

```

结果如下：





##### 3.4预测某商品时通过考虑商品热度进行衰减

```python
def prdictRating(user,item):
    c = len(UserPerItem[item])#计算有多少个用户评价过该商品
    c = 1 / (1 + math.log(c,10))
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['business_id']
        if i2 == item:continue
        ratings.append(d['stars'])
        similarities.append(cos(UserPerItem[item],UserPerItem[i2]) * c)
    if(sum(similarities)>0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean
```

```python
#导入新测试集数据并预测
#读取测试集数据
filename2 = "test2.csv"
f2 = open(filename2, "rt", encoding="utf-8")
header2 = f2.readline()
header2 = header2.strip().split(',')#列名
#print(header2)
header2.append('pre_stars')
#print(header2)

predata = []

for line in f2:
    fields = line.strip().split(',')
    d = dict(zip(header2, fields))
    d["pre_stars"] = 0
    predata.append(d)

for d in predata:
    u, i = d['user_id'], d['business_id']
    s = round(prdictRating(u,i))
    d['pre_stars'] = s

df = pd.DataFrame(predata)
df.to_csv(filename2)
```



##### 3.5对比test.csv和test2.csv

```python
filename_1 = "test.csv"
filename_2 = "test2.csv"

file1 = open(filename_1, "rt", encoding="utf-8")
file2 = open(filename_2, "rt", encoding="utf-8")

headers1 = file1.readline()
headers1 = headers1.strip().split(',')#列名
headers2 = file2.readline()
headers2 = headers2.strip().split(',')#列名

data1 = []
data2 = []

for line in file1:
    fields = line.strip().split(',')
    d = dict(zip(headers1, fields))
    data1.append(d)

for line in file2:
    fields = line.strip().split(',')
    d = dict(zip(headers2, fields))
    data2.append(d)

s = len(data1)#总数据量
count = 0#统计评分相等的个数
for i in range(len(data1)):
    s1 = float(data1[i]['pre_stars'])
    s2 = float(data2[i]['pre_stars'])
    if(s1 == s2):
        count += 1
print(count / s)
```

<img src="/Users/lee/Study/大三-下/智能推荐系统/Project_1/lab1.assets/截屏2021-04-03 上午11.17.24.png" alt="截屏2021-04-03 上午11.17.24" style="zoom:50%;" />

### 4、提交文件列表

实验报告lab2-10185102142-李泽浩

源代码source_code.ipynb

预测结果后的test.tsv

