# 决策树

决策树是一种基本的分类与回归方法。决策树的模型呈树形结构，在分类模型中，表示基于特征对实例进行分类的过程。它可以被认为是if-then的规则集合，也可以认为是定义在特征空间与类空间上的条件概率分布。主要优点为模型具有可读性，分类速度快。学习时，利用训练数据，根据最小化损失函数的原则建立决策树模型，预测时，对新的数据，用决策树模型进行分类，学习通常包括了3个步骤：特征选择、决策树的生成和决策树的剪枝。该模型典型算法包括ID3算法、C4.5算法和CART算法，其中ID3算法使用信息增益来对特征进行选取，而C4.5算法则使用信息增益比率来选择特征，CART算法（Classification and regression tree分类回归树）则使用基尼指数来选择特征。


[决策树（一）](http://leijun00.github.io/2014/09/decision-tree/)

[决策树（二）](http://leijun00.github.io/2014/10/decision-tree-2/)

## 测试

使用sklearn中tree模块下的DecisionTreeClassifier，其中criterion可选gini和entropy表示通过gini系数和信息增益选取特征。

[1.10. 决策树](http://sklearn.apachecn.org/cn/stable/modules/tree.html)
[使用决策树处理iris数据集](http://www.letiantian.me/2015-03-31-decision-tree-iris/)

## 实现

[西瓜数据集2.0](../data/decision_tree/watermelon_2_0.txt)

```
编号,色泽,根蒂,敲声,纹理,脐部,触感,好瓜
1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,是
```

[machine-learning](https://github.com/tz28/machine-learning) 数据集从这个仓库中收集而来。

[如何实现并应用决策树算法？](http://whatbeg.com/2016/04/23/decisiontree.html) 实现思路主要参考这篇博客。

[决策树的实现](https://www.kancloud.cn/digest/machinglearninginact/102857) 增加参考。
