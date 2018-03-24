# 逻辑回归

逻辑斯蒂回归logistic regression是统计学习中的经典分类方法，该模型属于对数线性模型。

## 逻辑斯蒂分布
逻辑斯蒂分布：设$X$是连续随机变量，$X$服从逻辑斯蒂分布是指$X$具有下列分布函数和密度函数：
$$F(x)=\frac{1}{1+e^{-(x-\mu) / \gamma)}}$$
$$f(x) = \frac{e^{-(x-\mu) / \gamma)}}{\gamma (1+e^{-(x-\mu) / \gamma)})^2}$$
分布函数属于逻辑斯蒂函数，其图形如下所示，该曲线以点$(\mu, \frac{1}{2})$为中心对称，在中心附近增长速度较快，两端增长较慢，形状参数$\gamma$值越小，曲线在中心附近增长越快，可以理解为斜率$\frac{1}{\gamma}$。
![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/logistic_plot.png)

## 二项逻辑斯蒂回归模型

逻辑斯蒂回归模型：二项逻辑斯蒂回归模型满足如下条件概率分布：

$$P(Y=1|x)=\frac{exp(wx + b)}{1 + exp(wx + b)}$$
$$P(Y=1|x)=\frac{1}{1 + exp(wx + b)}$$

$w$称为权值向量，$b$称为偏置，逻辑回归比较两个条件概率值的大小，标记为概率值较大的那一类。为了方便，将权值向量和输入向量加以扩充，仍记住$w$，$x$，即$w=(w^(1),w^(2),...,w^(n),b)^T$，$x=(x^(1),x^(2),...,x^(n),1)^T$，上述公式简化如下：

$$P(Y=1|x)=\frac{exp(wx)}{1 + exp(wx)}$$
$$P(Y=1|x)=\frac{1}{1 + exp(wx)}$$

一个事件的几率：该事件发生的概率与该事件不发生的概率的比值。如果事件发生的概率是$p$，那么该事件不发生的概率是$1-p$，那么这个事件的几率是$\frac{p}{1-p}$，该事件的对数几率为$logit(p)=log{\frac{p}{1-p}}$

代入得到下述式子，表示输出$Y=1$的对数几率是输入$x$的线性函数：
$$log{\frac{P(Y=1|x)}{1-P(Y=1|x)}=wx}$$


## 参数估计

使用极大似然估计方法来估计逻辑斯蒂回归模型的参数，得到逻辑斯蒂回归模型，设$P(Y=1|x)=\pi(x)$，$P(Y=0|x)=1-\pi(x)$，那么似然函数为$\prod_{i=1}^{N}[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}$，对数似然函数为：

![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/logistic_like.png)

[【机器学习算法系列之二】浅析Logistic Regression](https://chenrudan.github.io/blog/2016/01/09/logisticregression.html) 这篇博文总结地较好

[逻辑回归LR推导（sigmoid，损失函数，梯度，参数更新公式）](https://hyzhan.github.io/2017/05/23/2017-05-23-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92LR%E6%8E%A8%E5%AF%BC%EF%BC%88sigmoid%EF%BC%8C%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%8C%E6%A2%AF%E5%BA%A6%EF%BC%8C%E5%8F%82%E6%95%B0%E6%9B%B4%E6%96%B0%E5%85%AC%E5%BC%8F%EF%BC%89/)

[Logistic回归的梯度下降法推导](http://ziyuanjun.github.io/2016/01/21/Logistic%E5%9B%9E%E5%BD%92%E7%9A%84%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95%E6%8E%A8%E5%AF%BC/)

[LogisticRegression](https://github.com/perborgen/LogisticRegression/blob/master/logistic.py) 代码实现，根据推导公式而来，加深理解

[逻辑回归模型(Logistic Regression, LR)基础](https://www.cnblogs.com/sparkwen/p/3441197.html)

## 测试

[1.1.11. logistic 回归](http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#logistic)

## 实现





















