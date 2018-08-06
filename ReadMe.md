# 统计学习模型学习笔记

记录统计学习模型学习过程中的笔记，其中包括使用sklearn来测试模型，同时包括自己构建模型来实现模型的训练等等。


## 统计学系方法概论

## 感知机

## k近邻法

## k-means
增加k-means聚类算法。

## 朴素贝叶斯法

## 决策树

## 逻辑斯蒂回归与最大熵模型
其中增加线性回归的推导。

## 支持向量机

## 提升方法

### AdaBoost
增加AdaBoost方法。

## EM算法及其推广

## 隐马尔可夫模型

## 条件随机场

## 深度学习
额外增加神经网络这一章节补充神经网络相关知识，主要是BP误差反向传播算法的推导。

### CNN
细分深度学习分支之卷积神经网络。

### RNN
细分深度学习分支之循环神经网络。

### GAN
细分深度学习分支之生成对抗网络。

### VAE
variational auto-encoder变分自动编码器。

### Attention
增加Attention机制。

## 数据降维

### PCA
增加PCA。

### LDA
增加LDA。

## 激活函数
增加激活函数理解总结部分。

## 优化器

该章节主要介绍常见的优化器算法，将梯度下降和牛顿法都归并到该章节中，详细参考[优化器方法](optim/ReadMe.md)。

### 梯度下降
增加梯度下降解释理解部分，该部分比较了梯度下降和牛顿法，其中梯度下降优化曲线较牛顿法慢，呈“之”字形，但是牛顿法存在Hessian矩阵计算量大等问题。

## 牛顿法
增加牛顿法算法实现总结。

## 矩阵学习
增加矩阵学习相关，主要是矩阵、向量、标量求导。

## 步长搜索
增加步长搜索算法总结。

## 学习率
增加learning rates和schedules相关算法总结，具体参考[lr学习率学习](./lr/ReadMe.md)。

## 多项式拟合
使用多项式拟合来拟合数据。

## 深度学习框架
增加深度学习框架使用教程，包括Tensorflow、PyTorch等。

下述是其他深度学习框架：
- [DeepPy: Deep learning in Python](http://andersbll.github.io/deeppy-website/) 纯用numpy实现的深度学习框架（GPU加速）。

### TensoFlow
细分深度学习框架TensorFlow使用教程。

### PyTorch
细分深度学习框架PyTorch使用教程。

### MXNet
细分深度学习框架MXNet使用教程。

## 协方差矩阵
增加协方差矩阵计算。

## 线性代数

增加线性代数。

### 迹

参考[PyMathModule](https://github.com/guanfuchen/PyMathModule)仓库中矩阵之迹相关。

### SVD

增加SVD奇异值分解，参考[svd](./svd/ReadMe.md)


---
## 参考资料

[MachineLearning机器学习实战](https://github.com/apachecn/MachineLearning/tree/master/src/py2.x/ML) 机器学习实战包含了书籍和示例代码，其中包括了自我实现的以及sklearn中的常用算法示例，[机器学习实战](http://ml.apachecn.org/mlia/) 网页浏览。

[machine-learning-notes](https://github.com/roboticcam/machine-learning-notes) 徐亦达教授总结的latex书写的机器学习知识。

[Mathematics](https://github.com/Ewenwan/Mathematics) 作者记录的数学知识点仓库，结构思路可以参考，以及对应的[MVision计算机视觉相关](https://github.com/Ewenwan/MVision)。




