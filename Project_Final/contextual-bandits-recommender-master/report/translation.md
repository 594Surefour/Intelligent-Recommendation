## <center>LinUCB vs HybridLinUCB in recommender systems



### 1.Introduction

​      LinUCB和HybridLinUCB的算法都是在2012年发表的文章《A Contextual-Bandit Approach to Personalized News Article Recommendation》中介绍的。
​      作者决定将新闻文章的个性化推荐建模为情境匪徒问题。在这一原则性方法中，学习算法根据用户和文章的上下文信息依次选择文章为用户服务，同时调整其文章选择策略。

### 2.Theory

​       这两种算法都是基于置信度上限（UCB）的方法，使用一种更聪明的方式来平衡 勘探和开发，而不是简单的'epsilon-greedy'战略。 然后他们选择达到最高置信度上限的手臂。有了适当定义的置信区间，可以证明这种算法有一个很小的总T-试验遗憾，它只与总试验数T成对数，结果是最佳的。

#### 2.1LinUCB with Disjoint Linear Models (LinUCB)

​    我们假设一个手臂a的预期报酬在其d维特征xt,a中是线性的，有一些未知的系数向量θa∗，即：

<img src="translation.assets/截屏2021-05-31 下午7.13.40.png" alt="截屏2021-05-31 下午7.13.40" style="zoom:50%;" /><img src="translation.assets/截屏2021-05-31 下午7.14.00.png" alt="截屏2021-05-31 下午7.14.00" style="zoom:50%;" />

这个模型被称为离散模型，因为不同的手臂之间不共享参数。解决方案是通过一个简单的脊回归来实现的，图1中可以看到一个允许增量更新参数矩阵的算法。伪代码实现（具有增量更新参数矩阵的LinUCB算法）：

<img src="translation.assets/截屏2021-05-31 下午7.14.58.png" alt="截屏2021-05-31 下午7.14.58" style="zoom: 33%;" />

#### 2.2LinUCB with Hybrid Linear Models (HybridLin- UCB)

在许多应用中，除了特定手臂的功能外，使用所有手臂共享的功能也很有帮助。例如，在新闻文章推荐中，用户可能只喜欢有关政治的文章，这就提供了一个机制。因此，拥有共享和非共享部分的特征是有帮助的。从形式上看，我们通过在前面的方程中加入另一个线性项来采用以下的混合模型。

<img src="translation.assets/截屏2021-05-31 下午7.18.32.png" alt="截屏2021-05-31 下午7.18.32" style="zoom:50%;" />

### 3.Implementation

​        算法总是在选定的时间段内运行。一个历时意味着该算法为数据集中的每个用户迭代产生了一个建议。在每个历时中，用户的排序是随机的。当有多个可能的手臂可供选择时=它们的pt等于max(pt)，最后的手臂会从这些手臂中随机选择。伪代码（具有增量更新参数矩阵的混合LinUCB算法）：
<img src="translation.assets/截屏2021-05-31 下午7.22.36.png" alt="截屏2021-05-31 下午7.22.36" style="zoom:50%;" />



### 4.Experiments

#### 4.1 Data Preprocessing

​      我决定使用MovieLens的100k数据集。它有100 000个评分，1000个用户和1700部电影。[3]
​      由于计算上的要求，我从整个数据集中抽取了一个小的子集，包含100个项目和56个用户。为了确保所有的用户都至少有一些评分，我给每个用户随机添加了3个评分。
​      按照推荐领域的惯例，我将评分从1-5分的范围内二进制化为
​			1 = 积极预测 = 预测值4或者更高
​			-1 = 消极预测 = 预测值小于4
​			0 = 位置预测
我决定将负面评价与未知评价分开，因为我需要它们来模拟未知用户的评价。

#### 4.2Modeling the user behavior

​       这篇论文描述了一种复杂的方法来评估离线算法，这很有趣，但这样的程序不在本项目的范围之内。因此，我决定使用一个简单的用户模型来预测一个未见过的项目的正面或负面评价。
​       如果用户u已经对项目a进行了评价，那么返回的奖励将是1，即正面评价或0，即负面评价。可以用概率为1/0的伯努利分布代替这些固定值来计算奖励。
​        如果用户u还没有对项目i进行评分，奖励将从伯努利分布中抽样，p等于用户对项目i的流派的喜欢程度。对一个流派ga的喜欢程度被计算为属于流派ga的项目的正面评分与属于ga的项目的负面评分的比率。只有用户u的评分被计算在内。如果项目i属于多个流派，那么所得的喜好度将被计算为所有项目i流派喜好度的平均值。

#### 4.3 Experiment 1: Fixed rewards

在这个实验中，我简单地将已实现的算法与前面几章中描述的设置一起运行了50个epochs。
<img src="translation.assets/截屏2021-05-31 下午7.38.17.png" alt="截屏2021-05-31 下午7.38.17" style="zoom:50%;" />

​      可以看出，LinUCB很快就达到了开发阶段，这是特别容易的，因为它只需要为每个用户找到一个返回正奖励的项目，然后一直选择它。
​      另一方面，HybridLinUCB正在探索更多的东西，因此，每个历时的平均奖励增加得较慢，但最终取得了类似的结果。

#### 4.4 Experiment 2:Stochastic rewards

​       在这个实验中，我修改了对已经评级的项目给予奖励时的用户模型行为。如果之前的评价是正面的，奖励将从伯努利分布中取样，p = 0.9，否则p = 0.1。所有其他的参数都保持不变。
<img src="translation.assets/截屏2021-05-31 下午7.39.26.png" alt="截屏2021-05-31 下午7.39.26" style="zoom:50%;" />

​       LinUCB再次试图快速开发，但它的进展受到了噪音的阻碍，尽管如此，它最终还是设法保持在接近最佳的0.85附近。
​      HybridLinUCB再次进行了更多的探索，而嘈杂的奖励使它更加缓慢，这就是为什么我让它运行了100个历时。可以看出，它最终在稍差的0.8的平均奖励附近震荡。





## translations in readme

MovieLens数据集是由明尼苏达大学的GroupLens研究项目收集的。
在明尼苏达大学收集的。
这个数据集由以下内容组成。
 * 来自943个用户对1682部电影的100,000个评分（1-5）。 
 * 每个用户至少对20部电影进行了评分。
 * 用户的简单人口统计信息（年龄、性别、职业、邮编）。

这些数据是通过MovieLens网站从1997年9月19日到1998年4月22日的七个月期间，收集的(movielens.umn.edu)收集的，评分少于20次或没有完整的人口统计资料的用户被从该数据集中删除。

