# 感知机

感知机是一个二分类线性分类模型，输入实例的特征向量，输出实例的类别（+1，-1），旨在求出将训练数据进行线性划分的分离超平面，属于判别模型，其分类误差为误分类的损失函数，利用梯度下降法对损失函数进行极小化，求得感知机模型。
模型定义：

$$f(x)=sign(wx+b)$$

[《统计学习方法》读书笔记——感知机](http://www.cnblogs.com/OldPanda/archive/2013/04/12/3017100.html)

[神经网络入门](http://www.ruanyifeng.com/blog/2017/07/neural-network.html)

## 收敛性证明
对于线性可分数据集，感知器学习算法原始形式收敛，即经过有限次迭代可以得到一个将训练数据集完全正确划分的分离超平面及感知机模型。
令$\hat{w}=(w^T,b)^T$，$\hat{x}=(x^T,1)^T$，那么$\hat{x} \in R^{n+1}$，$\hat{w} \in R^{n+1}$，原始感知器模型$f(x)=sign(w \cdot x + b)$转变为$f(x) = sign(\hat{w} \cdot \hat{x})$
Novikoff定理：设训练数据集$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$是线性可分的，其中$x_i \in X = R^n$，$y_i \in y = \{-1, +1\}$，则
（1）存在满足条件$||\hat{w_{opt}}||=1$的超平面$\hat{w_{opt}} \cdot \hat{x} = w_{opt} \cdot x + b_{opt}$将训练集完全分开，且存在$\gamma>0$对所有$i=1,2,...,N$，满足$y_i (\hat{w_{opt}} \cdot \hat{x_i}) = y_i (w_{opt} \cdot x_i + b_{opt}) \geq min(y_i (w_{opt} \cdot x_i + b_{opt}))$
（2）令$R=\max_{1\leq i \leq N}{||\hat{x_i}||}$，则感知器算法在训练集上的误分类次数$k$满足不等式：
$$k \leq (\frac{R}{\gamma})^2$$

证明如下两个不等式即可：
$$\hat{w_k} \cdot \hat{w_{opt}} \geq k \eta \gamma$$
$$||\hat{w_k}||^2 \leq k {\eta}^2 {R}^2$$
根据以上两个不等式：
推出：
$$k \eta \gamma \leq \hat{w_k} \cdot \hat{w_{opt}} \leq ||\hat{w_k}|| ||\hat{w_{opt}}|| \leq \sqrt{k} \eta R$$
$$k^2 {\gamma}^2 \leq k R^2$$
于是，下述不等式得证：
$$k \leq (\frac{R}{\gamma})^2$$
该定理表明，误分类次数$k$是有上届的，经过有限次搜索可以找到将训练数据完全分开的分离超平面。

第一个不等式证明：
$$k \eta \gamma \leq \hat{w_k} \cdot \hat{w_{opt}} \leq ||\hat{w_k}|| ||\hat{w_{opt}}|| \leq \sqrt{k} \eta R$$
对于第k-1次误分类，下一次的权重更新为$\hat{w_k}=\hat{w_{k-1}}+\eta y_i \hat{x_i}$，那么$\hat{w_k} \cdot \hat{w_{opt}}=\hat{w_{k-1}} \cdot \hat{w_{opt}}+\eta y_i (\hat{w_{opt}} \cdot \hat{x_i}) \geq \hat{w_{k-1}} \cdot \hat{w_{opt}} + \eta \gamma \geq \hat{w_{k-2}} \cdot \hat{w_{opt}} + 2 \eta \gamma \geq ... \hat{w_{0}} \cdot \hat{w_{opt}} + k \eta \gamma = k \eta \gamma$

第二个不等式证明：
$$k^2 {\gamma}^2 \leq k R^2$$
$$||\hat{w_k}||^2 = ||\hat{w_{k-1}}+\eta y_i \hat{x_i}||^2 = ||\hat{w_{k-1}}||^2 + 2 \eta y_i \hat{w_{k-1}}\hat{x_i}+{\eta}^2 ||\hat{w_i}||^2 \leq ||\hat{w_{k-1}}||^2 + {\eta}^2 R^2 \leq ... \leq k {\eta}^2 R^2$$

[怎么理解感知器（perceptron）?](https://www.zhihu.com/question/21636295/answer/286999321?utm_source=com.android.email&utm_medium=social)

## 测试

使用sklearn对感知机模型进行测试，了解其性能和训练过程。

![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/perceptron_sklearn.png)

[1.1.13. Perceptron感知器](http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#perceptron)

[利用sklearn学习《统计学习方法》（一）：感知机（perceptron）](https://zhuanlan.zhihu.com/p/27152953)

## 实现

主要使用感知机的原始形式进行学习，对于误分类的样本，使用下述公式进行sgd更新。

$$w = w + \eta * y_i * x_i$$

$$b = b + \eta * y_i$$

[A Perceptron in just a few Lines of Python Code](https://maviccprp.github.io/a-perceptron-in-just-a-few-lines-of-python-code/)
