# 深度学习

## 梯度下降算法

## 反向传播算法

神经网络使用梯度下降算法来学习权重和偏置，梯度的计算则是使用反向传播算法。反向传播算法的核心是对一个代价函数$C$关于任何权重$w$或者偏置$b$的偏导数$\partial{C} / \partial{w}$或者$\partial{C} / \partial{b}$。
给出BP的四个基本公式前，首先对用到的符号解释
- 权重$w_{jk}^{l}$表示从第$(l-1)$层的第$k$个神经元到第$l$层的第$j$个神经元的链接上的权重。
- 偏置$b_j^l$表示在第$l$层第$j$个神经元的偏置
- 激活值$a_j^l$表示在第$l$层第$j$个神经元的激活值
- 带权输入，第$l$层神经网络激活前的值$z^l$，其中$z_j^l$表示第$l$层第$j$个神经元的激活函数的带权输入
- 误差，第$l$层网络的误差标记为$\delta^l$，$\delta_j^l$表示第$l$层第$j$个神经元上的误差，实际上就是代价函数对对应带权输入的偏导数$\partial{C} / \partial{z_j^l}$

![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/layer_weight_symbol.png)
![](http://chenguanfuqq.gitee.io/tuquan/bias_symbol.png)

神经网络第$l$层的第$j$个神经元的激活值$a_j^l$计算如下所示：
$$a_j^l=\sigma(\sum_{k}{w_{jk}^l a_j^{l-1} + b_j^l})$$
矩阵表示为
$$a^l=\sigma(w^l a^{l-1} + b^l)$$
带权输入和激活值的关系
$$a^l=\sigma(z^l)$$
$$z^l=w^l a^{l-1} + b^l$$
$$z_j^l=\sum_{k}{w_{jk}^l a_j^{l-1} + b_j^l}$$
误差和带权输入的关系
$$\delta_j^l = \partial{C} / \partial{z_j^l}$$

通常称$\delta_l$为$l$层的误差向量，BP会提供一种计算每一层$\delta^l$的方法，然后将这些误差和最终的$\partial{C} / \partial{w_jk^l}$或者$\partial{C} / \partial{b_j^l}$联系起来。其中BP算法的四个基本方程就是阐述了这个关系：
- BP1 **输出层误差的方程**
$$\delta^L = \nabla_{a}{C} \odot \sigma^{\prime}(z^L)$$

- BP2 **使用下一层的误差表示当前层的误差**
$$\delta^l = ((w^{l+1})^T \delta^{l+1})) \odot \delta^{\prime}(z^l)$$

- BP3 **代价函数关于网络中任意偏置的改变率**
$$\frac{\partial{C}}{\partial{b_j^l}} = \delta_j^l$$

- BP4 X**代价函数关于任何一个权重的改变率**
$$\frac{\partial{C}}{\partial{w_{jk}}^l} = a_k^{l-1} \delta_j^l$$

BP1和BP2可以计算每一层网络的误差，BP3和BP4通过误差和权重、偏置的关系计算每一个相应的梯度
具体的推导如下所示：
### BP1推导
BP1的推导较为简单，给出了求误差的初始值
$$\delta^L = \nabla_{a}{C} \odot \sigma^{\prime}(z^L)$$
$$\delta_j^L = \frac{\partial{C}}{\partial{z_j^L}}  = \frac{\partial{C}}{\partial{a_j^L}} \frac{\partial{a_j^L}}{\partial{z_j^L}} = \frac{\partial{C}}{\partial{a_j^L}} \sigma^{\prime}(z_j^L)$$

### BP2推导
BP2给出了当前层的误差和上一层误差的关系，再根据最后一层的误差值，可递归获取所有层的误差
$$\delta^l = ((w^{l+1})^T \delta^{l+1})) \odot \delta^{\prime}(z^l)$$
$$\delta_j^l = \frac{\partial{C}}{\partial{z_j^l}} = \sum_{k}{\frac{\partial{C}}{\partial{z_k^{l+1}}} \frac{\partial{z_k^{l+1}}}{\partial{z_j^{l}}}} = \sum_{k}{\frac{\partial{z_k^{l+1}}}{\partial{z_j^{l}}} \delta_k^{l+1}}$$

对下述式子做微分可得$\frac{\partial{z_k^{l+1}}}{\partial{z_j^{l}}}$
$$z_k^{l+1} = \sum_{j}{w_{kj}^{l+1} a_j^l + b_k^{l+1}} = \sum_{j}{w_{kj}^{l+1} \sigma(z_j^l) + b_k^{l+1}}$$
$$\frac{\partial{z_k^{l+1}}}{\partial{z_j^{l}}} = w_{kj}^{l+1} \sigma^{\prime}(z_j^l)$$
带入可得BP2:
$$\delta_j^l = \sum_{k}{ w_{kj}^{l+1} \sigma^{\prime}(z_j^l) \delta_k^{l+1}}$$

### BP3推导

$$\frac{\partial{C}}{\partial{b_j^l}} = \delta_j^l$$
$$\frac{\partial{C}}{\partial{b_j^l}} = \frac{\partial{C}}{\partial{z_j^l}} \frac{\partial{z_j^l}}{\partial{b_j^l}} = \delta_j^l$$

### BP4推导
$$\frac{\partial{C}}{\partial{w_{jk}^l}} = a_k^{l-1} \delta_j^l$$

$$\frac{\partial{C}}{\partial{w_{jk}^l}} = \frac{\partial{C}}{\partial{z_{j}}^l} \frac{\partial{z_{j}}^l}{\partial{w_{jk}^l}}=a_k^{l-1} \delta_j^l$$



### 代价函数C
为了使用BP算法，代价函数$C$满足两个前提假设
- 代价函数可以被写成一个在每一个训练样本$x$上的代价函数$C_x$的均值$C=\frac{1}{n} \sum_{x} C_x$
- 代价函数可以写成神经网络输出的函数

### Hadamard乘积
Hadamard乘积是向量按照元素乘积，即$(s \odot t)_j = s_j t_j$

![](http://chenguanfuqq.gitee.io/tuquan/hadamard.png)

参考《神经网络与深度学习》

[Deep Learning深度学习（二）：反向传播](http://chansh518.github.io/deep%20learning/2016/08/08/Deep-Learning-Notes-Backpropagation.html)

[神经网络入门](http://www.ruanyifeng.com/blog/2017/07/neural-network.html)

[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)

[neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning)

[《解析卷积神经网络—深度学习实践手册》](http://lamda.nju.edu.cn/weixs/book/CNN_book.html) 魏秀参老师对于深度学习实践的总结，非常值得学习

[Must Know Tips/Tricks in Deep Neural Networks (by Xiu-Shen Wei)](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)

[深度学习与神经网络全局概览：核心技术的发展历程](https://www.jiqizhixin.com/articles/2016-08-08-2)

[机器学习公开课笔记(4)：神经网络(Neural Network)——表示](http://www.cnblogs.com/python27/p/MachineLearningWeek04.html)

## CNN反向传播

[卷积神经网络中卷积计算，卷积核的旋转？](https://www.zhihu.com/question/55015134)

## 测试

## 实现














