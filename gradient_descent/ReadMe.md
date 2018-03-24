# 梯度下降

梯度下降是一种求解最优化的方法，假设我们要最小化某些函数，$C(v)$。它可以是任意的多元实值函数，$v=v_1,v_2,...$。为了最小化C(v)，想象C是一个只有两个变量$v_1$和$v_2$的函数，一种找到C的全局最小值的方法是可以计算导数来寻找C的极值点，这对于求解只有一个或者少数几个变量的函数来说可行，但是如果变量过多那就不切实际（求解导数为0变得较慢）。在神经网络中使用了大量的权重和偏置等参数，极其复杂，因此通过微积分的方法来计算最小值变得不可行。
首先把我们的优化函数想象成一个山谷，一个小球从山谷的斜坡滚落下来，经验告诉我们这个球最终会滚到谷底，梯度下降就是利用这一想法来找到函数的最小值。我们为一个球体随机选择一个起始位置（初始化的$w_0$和$b_0$），然后模拟球体滚落到谷底的运动。
这里将问题重新描述一下，即如何在$v_1$和$v_2$方向移动一个很小的量，即$\Delta v_1$和$\Delta v_2$时，球体会下降，首先观察改变后球体位置的变化$\Delta C$：
$$\Delta C \approx \frac{\partial{C}}{\partial{v_1}} \Delta{v_1} + \frac{\partial{C}}{\partial{v_2}} \Delta{v_2}$$
我们要寻找一种选择$\Delta{v_1}$和$\Delta{v_2}$的方法使得$\Delta C$为负。定义$\Delta v = (\Delta v_1, \Delta v_2)^T$和$\nabla C = (\frac{\partial{C}}{\partial{v_1}}, \frac{\partial{C}}{\partial{v_2}})^T$，其中$\nabla C$表示梯度向量，也就是$C$的偏导数向量。
那么，上述优化式子改写为：
$$\Delta C \approx \nabla C \cdot \Delta v$$
这个方程给出了如何取$\Delta v$使得$\Delta C$为负数，假设我们选取：
$$\Delta v = -\eta \nabla C$$
这里的$\eta$是一个很小的正数（神经网络中称为学习率），那么$\Delta C \approx \nabla C \cdot \Delta v = \eta \nabla C \cdot \nabla C = -\eta ||\nabla C||^2$，这个式子保证了$\Delta C \leq 0$，所以根据这个规则不断改变$v$，C会一直减小，不会增加（当然要在$\Delta C$的近似约束下才能成立）。这里我们总结这种方法为梯度下降算法：
$$v \rightarrow v^{\prime} = v - \eta \nabla C$$
上述式子中的$\eta$是学习率，不能过大也不能过小。过大会导致球体滚落急剧变化，学习易震荡，而过小的学习率会导致球体滚落较慢，学习非常缓慢。深度学习中会在学习的过程中调整学习率，首先给一个较大的学习率使得快速学习下降到极值点附近，然后减小学习率（比如初始学习了0.1，经过50个epoch改变为0.01，然后经过50个epoch改变为0.001等等），使得不断接近最小值。

![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/2018-03-21_2.26.53.png)

[四天速成！香港科技大学 PyTorch 课件分享](https://www.jiqizhixin.com/articles/2017-10-09-4)

[梯度下降（Gradient Descent）小结](https://www.cnblogs.com/pinard/p/5970503.html)

[理解梯度下降](http://liuchengxu.org/blog-cn/posts/dive-into-gradient-decent/)

[神经网络 梯度下降](https://morvanzhou.github.io/tutorials/machine-learning/torch/1-1-C-gradient-descent/)

[PyTorch: 梯度下降及反向传播](http://blog.csdn.net/m0_37306360/article/details/79307354)

## 测试

## 实现











