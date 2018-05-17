# PCA

主成分分析（Principal Component Analysis，简称PCA）是最常用的一种降维方法。在介绍PCA之前，不妨先考虑这样一个问题：对于正交属性空间中的样本点，如何用一个超平面对所有样本进行恰当的表达？

容易想到，若存在这样的超平面，那么它大概应具有这样的性质：
- 最近重构性：样本点到这个超平面的距离足够近；
- 最大可分性：样本点在这个超平面上的投影能尽可能分开。
基于最近重构性和最大可分性，可以得到主成分分析的两种等价推导。

假设在$R^n$空间中我们有$m$个点$\{x^{(1)},\cdots,x^{(m)}\}$，我们希望对这些点进行有损压缩。有损压缩表示我们使用更少的内存，但损失一些精度去存储这些点。我们希望损失的精度尽可能少。

一种编码这些点的方式使用底维表示。对于每个点$x^{(i)} \in R^n$，会有一个对应的编码向量$c^{(i)} \in R^l$。如果$l$比$n$少，那么我们便是用了更少的内存来存储原来的点。我们希望找到一个编码函数$f(x)$，根据输入返回编码，$f(x)=c$同时也希望找到一个解码函数$g(c)$，给定编码重构输入，$x \approx \hat{x}=g(c)=g(f(x))$。

以上便是PCA要解决的问题，那么如何找到这个编码函数和解码函数成为问题的关键。这里为了简化解码器，使用矩阵乘法将编码映射回$R^n$，即$g(c)=Dc，其中D \in R^{nxl}是定义解码的矩阵$。

上述如果$D$和$c$等比例缩小和放大，那么PCA得到的解存在无穷多个。这里限制$D$中所有列向量都有单位范数来使问题有唯一解。

另外为了使编码问题简化，PCA同时限制了$D$的所有列向量彼此正交。

我们希望损失的精度尽可能少，那么定义优化目标函数为最小化原始输入向量$x$和重构向量$g(c)$之间的距离，这里使用二范数表示这个距离。

$$\bar{c}=\arg \min_{c}{||x-g(c)||_2}$$
$$\bar{c}=\arg \min_{c}{(x-g(c))^T(x-g(c))}$$
最小化函数简化为：
$$x^T x - x^T g(c) - g(c)^T x + g(c)^Tg(c)$$
$g(c)^T x$为标量，其转置和本身相同。
$$x^T x - 2 x^T g(c) + g(c)^Tg(c)$$
将$g(c)=Dc$带入，同时去除无关项$x^T x $，得到：
$$- 2 x^T D c + c^T D^T D c$$
根据以上对解码器的两个限制，$D^T D=I_l$，代入得到：
$$- 2 x^T D c + c^T c$$
那么对这个优化函数求导（这里用到了矩阵的求导法则）：
$$\nabla_c{- 2 x^T D c + c^T c}=-2 D^T x + 2c=0$$
那么：
$$c=D^T x$$
从这里我们可以看出，为了使得压缩损失越小，编码函数$f(x)=D^T x$，那么编码和解码都这需要一个矩阵向量乘法操作，使得算法非常高效，下述为经过编码解码后重构的向量：
$$r(x)=g(c)=g(f(x))=D D^T x$$
那么问题化为求解解码矩阵$D$，这里对所有样本的重构误差作为优化目标函数进行最小化重构误差。
$$\hat{D}=\arg \min_{D} \sqrt{\sum_{i,j}{(x_j^{(i)}--r(x_{i})_j)^2}} 在D^T D =I_l约束下$$
这里为了推导用于寻求$\hat{D}$的算法，首先考虑$l=1$的情况，即最后压缩为一维向量，这种情况下，$D$是一个单一的向量$d \in R^{nx1}$，上式化简为：
$$\hat{d}=\arg \min_{d} \sqrt{\sum_{i,j}{(x_j^{(i)}--r(x_{i})_j)^2}} 在d^T d =1约束下$$
这里直接进行矩阵化简，使用$X \in R^{mxn}$表示特征样本矩阵。
$$\hat{d}=\arg \min_{d} {||X-X d d^T||_F^2} 在d^T d =1约束下$$
其中F范数计算为：
$${||X-X d d^T||_F^2}=tr((X-X d d^T)^T (X-X d d^T))$$
展开为：
$$tr(X^T X-X^T X d d^T - d d^T X^T X + d d^T X^T X d d^T)$$
将$d^T d =1$代入同时去除无关项：
$$\arg \min_{d} tr(-X^T X d d^T - d d^T X^T X + d d^T X^T X d d^T)$$
$$\arg \min_{d} -tr(X^T X d d^T)-tr(d d^T X^T X) + tr(d d^T X^T X d d^T)$$
$$\arg \min_{d} - tr(X^T X d d^T)-tr(d d^T X^T X) + tr(X^T X d d^T d d^T)$$
$$\arg \min_{d} - 2 tr(X^T X d d^T) + tr(X^T X d d^T)$$
$$\arg \min_{d} - tr(X^T X d d^T)$$
$$\arg \max_{d} tr(X^T X d d^T)$$
通过拉格朗日乘子法构造$L(d)= tr(X^T X d d^T)+\lambda (1-d^T d)$，对$d$求导数得到：
$$\frac{\partial{L(d)}}{\partial{d}}=2X^T X d -2 \lambda d=0$$
则$X^T X d = \lambda d$，同时$$\arg \max_{d} tr(X^T X d d^T)=\arg \max_{d} tr(\lambda d d^T)=\arg \max_{d} tr(\lambda d^T d)=\lambda$$
得到$X^T X d = \lambda d$，那么$d$是$X^T X$最大特征值对应的特征向量。通过归纳法可得$l>1$也是相同的证明做成，矩阵D是由前$l$个最大的特征值对应的特征向量组成。

---
## 参考资料

[Machine Learning小结(4)：主成分分析（PCA）](http://blog.kongfy.com/2014/11/machine-learning%E5%B0%8F%E7%BB%934%EF%BC%9A%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90%EF%BC%88pca%EF%BC%89/) 作者总结了Ng课程中关于PCA的相关总结。

[Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) 维基百科中关于PCA的解释，[主成分分析 维基百科中文](https://zh.wikipedia.org/zh-hans/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)。

[Singular-value decomposition](https://en.wikipedia.org/wiki/Singular-value_decomposition) 维基百科中关于SVD的解释，PCA算法中使用的协方差矩阵SVD分解。

[PCA算法浅析](http://jermmy.xyz/2017/03/25/2017-3-25-understand-PCA/) 作者讲解了PCA算法流程。

[主成分分析（Principal components analysis）-最大方差解释](http://www.cnblogs.com/jerrylead/archive/2011/04/18/2020209.html)

[Priniciple Component Analysis (and various Eigen-things)](https://courses.cs.washington.edu/courses/cse446/18wi/sections/section4/PCA_Notebook.html#Priniciple-Component-Analysis-(and-various-Eigen-things))

[协方差矩阵](http://jermmy.xyz/2017/03/19/2017-3-19-covariance-matrix/) 其中介绍了协方差矩阵的计算。

[用scikit-learn学习主成分分析(PCA)](https://www.cnblogs.com/pinard/p/6243025.html)

[PCA的数学原理及推导证明](https://zhuanlan.zhihu.com/p/26951643) 基本是深度学习中的内容，原内容参考深度学习中文版，2.212实例：主成分分析。

[PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html) 其中有实例解释，形象。

[主成分分析(PCA)原理及推导](https://blog.csdn.net/zhongkejingwang/article/details/42264479)
