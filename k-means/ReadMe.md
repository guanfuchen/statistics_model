# k-means
聚类是一种无监督的学习，它将相似的对象归到同一个聚类簇中。聚类方法几乎可以应用与所有对象，簇内的对象越相似，聚类的效果越好。其中k-means，即k-均值算法，可以发现k个不同的簇，且每一个簇的中心采用簇中所含值的均值计算而成。
首先介绍簇识别（cluster identification），簇识别给出聚类结果的含义。假定有一些数据，现在将相似数据归到一起，簇识别会告诉我们这些簇是什么。聚类与分类最大的不同在于，分类的目标事先已知，而聚类则不一样。因为其产生的结果与分类相同，而知识类别没有预先定义，聚类有时也被称为务监督分类。

聚类分析试图将相似对象归入同一簇中，将不相似对象归到不同簇。相似这一概念取决于所选择的相似度计算方法。

---
## 算法详解
- 优点：容易实现；
- 缺点：可能收敛到局部最小值，在大规模数据集上收敛较慢；
- 使用数据类型：数值型数据。

k-均值是发现给定数据集的k个簇的算法。簇个数k是用户给定的，每一个簇通过其质心（centroid），即簇中所有点的中心来描述。算法的流程为，首先，随机确定k个初始点作为质心；然后将数据集中的每一个点分配到一个簇中，具体来讲，为每一个点找距离其最近的质心，并将其分配给该质心所对应的簇；最后每一个簇的质心更新为该簇所有点的平均值；循环迭代直至簇质心不在变化。
上述提到最近质心通过某种距离计算。

评价指标：

使各个样本与所在簇的质心的均值的误差平方和达到最小（这也是评价K-means算法最后聚类效果的评价标准）。

$$SSE=\sum_{i=1}^{k}\sum_{x \in C_i}|x-\mu_i|^2$$

---
## 应用场景
k-means可以应用在无监督学习中，通过聚类学习到特征，具体案例如图像分割，将纹理相同的图像分割为同一类别，如下图所示：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/screen_2018-05-10_15.38.32.png)

---
## 参考资料

《机器学习实战》第10章，利用K-均值聚类算法对未标注数据分组。

[k-Means算法](https://jhljx.github.io/2018/01/30/kMeans/)

[sklearn.cluster.KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) sklearn中k-means文档。

[k-平均算法](https://g.luciaz.me/extdomains/zh.wikipedia.org/zh-hans/K-%E5%B9%B3%E5%9D%87%E7%AE%97%E6%B3%95) k-means维基百科资料。

[K-means聚类算法](http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006910.html)

[MachineLearning](https://github.com/csuldw/MachineLearning/tree/master/Kmeans) 其中包含了较多的机器学习示例。

[机器学习算法-K-means聚类](http://www.csuldw.com/2015/06/03/2015-06-03-ml-algorithm-K-means/) 介绍k-means方法较好。

可参考Clustering Lecture 14中的PPT，David Sontag。

[scipy.cluster.vq.kmeans](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html) scipy中kmeans算法。
