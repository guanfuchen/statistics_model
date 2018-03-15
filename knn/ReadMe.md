# k近邻法

k近邻法是一种基本分类与回归算法，利用训练集对特征向量进行划分，实际上是利用训练数据集对特征向量空间进行划分并作为分类的模型。可直观理解为：给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的k个实例，这k个实例的多例属于某一个类，就把该输入实例分为这个类。

![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/kd_tree.png)

# 测试

[机器学习实战 之 kNN 分类](https://zhuanlan.zhihu.com/p/23191325)
[【量化课堂】一只兔子帮你理解 kNN](https://www.joinquant.com/post/2227?f=zh)

# 实现

实现过程使用欧氏距离度量相似性，获取k个邻近值后，分类决策规则使用多数表决（由k个邻近的训练实例中的多数类决定输入实例的类）

- 输入：X_train，y_train，测试数据X_test
- 输出：测试数据标签y_test
- 相似性度量def calc_similar(x1, x2)
- k邻近值计算def get_neighbors(X, y, k)
- 多数表决def get_decide(neighbors)