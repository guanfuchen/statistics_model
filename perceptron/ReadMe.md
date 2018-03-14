# 感知机

感知机是一个二分类线性分类模型，输入实例的特征向量，输出实例的类别（+1，-1），旨在求出将训练数据进行线性划分的分离超平面，属于判别模型，其分类误差为误分类的损失函数，利用梯度下降法对损失函数进行极小化，求得感知机模型。
模型定义：

$$f(x)=sign(wx+b)$$

## 测试

使用sklearn对感知机模型进行测试，了解其性能和训练过程。

![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/perceptron_sklearn.png)

[1.1.13. Perceptron（感知器](http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#perceptron)
[利用sklearn学习《统计学习方法》（一）：感知机（perceptron）](https://zhuanlan.zhihu.com/p/27152953)

## 实现

主要使用感知机的原始形式进行学习，对于误分类的样本，使用下述公式进行sgd更新。

$$w=w+\eta*y_i*x_i$$

$$b=b+\eta*y_i$$

[A Perceptron in just a few Lines of Python Code](https://maviccprp.github.io/a-perceptron-in-just-a-few-lines-of-python-code/)
