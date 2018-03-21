# 激活函数

激活函数层又称为非线性映射层，增加神经网络的表达能力（非线性能力）。否则，若干线性网络层的堆叠仍然只能起到线性映射的作用，无法形成复杂的函数。
直观上：激活函数模拟了生物神经元特性：接受一组输入信号并产生输出。在神经科学中，生物神经元通常有一个阈值，当神经元所获得的输入信号累积效果超过了该阈值，神经元就被激活而处于兴奋状态；否则处于抑制状态。在人工神经网络中，因Sigmoid型函数可以模拟这一生物过程，从而在神经网络发展历史进程中曾处于相当重要的地位。
Sigmoid型函数也称为Logistic函数：
$$\theta(x)=\frac{1}{1+\exp(-x)}$$
![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/2018-03-21_11.24.07.png)
经过Sigmoid函数作用后，输出响应的值域被压缩到$[0,1]$之间，而0对应了生物神经元的抑制状态，1则对应了兴奋状态。不过Sigmoid函数存在饱和问题，对于大于5或者小于5的值无论多大都会压缩到1或者0，这部分的梯度也接近0，在误差反向传播中导数处于该区域的误差将很难甚至根本无法传递到前层，进而导致整个网络无法训练。

为了避免Sigmoid函数中梯度饱和效应的发生，Hinton在2010年引入ReLU修正线性单元到神经网络中。
ReLU函数实际上是一个分段函数：
$$ReLU(x)=max(0,x)=\begin{cases}
x& if& x \geq 0\\
0& if& x < 0
\end{cases}$$
ReLU函数的梯度在$x \geq 0$时为1，反之为0。对$x \geq 0$部分完全消除了Sigmoid型函数的梯度饱和效应。同时，实验发现ReLU函数比Sigmoid型函数收敛速度更快，约6倍左右。正是由于ReLU函数的这些优质特性，ReLU函数已成为目前卷积神经网络及其他深度学习模型激活函数的首选之一。

![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/2018-03-21_11.41.25.png)

除了上述两种常见的激活函数外，接下来将系统介绍、对比七种当下深度卷积神经网络中国年常用的激活函数：Sigmoid函数、tanh（双曲正切）函数、修正线性单元（ReLU）、Leaky ReLU、参数化ReLU（PReLU）、随机化ReLU（RReLU）和指数化线性单元（ELU）

## Sigmoid函数
函数形式和基本介绍如上所述，另外从图中可以观察到Sigmoid函数值域的均值并非为0而是全为正，这不符合我们对神经网络内数值的期望均值应为0的设想。

## tanh函数
tanh型函数是在Sigmoid函数基础上为解决均值问题提出的激活函数：
$$tanh(x)=2\theta(2x)-1$$
tanh型函数又称为双曲正切函数，函数取值范围是$[-1,+1]$，输出响应的均值为0。可以观察到该函数依然会发生梯度饱和现象。

## ReLU函数
ReLU函数是目前深度卷积神经网络中最为常用的激活函数之一。
除了解决$x \geq 0$区间梯度消失的问题，同时ReLU整形线性单元相比Sigmoid函数和tanh函数，仅仅是和0的比较，以及线性操作，比指数函数的计算更为简单。但是对于$x < 0$这部分区间梯度为0，无法影响网络训练，这一线性称为死区。

## Leaky ReLU
为了缓解死区现象，研究者将ReLU函数中$x<0$的部分调整为$f(x)=\alpha \cdot x$，其中$\alpha$为0.01或者0.001数量级的较小正数。其中$\alpha=0$时Leaky ReLU函数退化为ReLU函数。但是$\alpha$的取值影响了Leaky ReLU的性能，合适的值较难设定且较为敏感，往往需要人为试验性地选取才能取得较好的性能。
$$Leaky ReLU(x)=max(0,x)=\begin{cases}
x& if& x \geq 0\\
\alpha \cdot x& if& x < 0
\end{cases}$$

## 参数化ReLU
参数化ReLU解决了Leaky ReLU函数中参数$\alpha$不易设定的问题，通过直接将$\alpha$作为一个网络中可学习的变量融入模型的整体训练过程求取。

## 随机化ReLU
随机化ReLU是随机化$\alpha$进行参数设定。

## 指数化线性单元

指数化线性单元ELU，ELU具备ReLU函数的优点，同时ELU也解决了ReLU函数自身的死区问题，不过ELU函数的指数操作稍稍增大了计算量，实际使用中，ELU中的超参数$\lambda$一般设置为1.

$$ELU(x)=\begin{cases}
x& if& x \geq 0\\
\lambda \cdot (\exp(x)-1)& if& x < 0
\end{cases}$$

![](http://chenguanfuqq.gitee.io/tuquan/img_2018_3/2018-03-21_1.08.28.png)

[DL 入门：常用激活函数及其应用](https://www.davex.pw/2017/10/11/activation-function/)

[Training Neural Networks, part I Activation functions, initialization, dropout, batch normalization](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf)

[几种常用激活函数介绍](http://wowx.info/posts/20160624/)

[The Activation Function in Deep Learning 浅谈深度学习中的激活函数](https://www.cnblogs.com/rgvb178/p/6055213.html)

[Activation function](https://en.wikipedia.org/wiki/Activation_function)

[26种神经网络激活函数可视化](https://www.jiqizhixin.com/articles/2017-10-10-3)

[激励函数 (Activation)](https://morvanzhou.github.io/tutorials/machine-learning/torch/2-03-activation/)

[非线性激活函数](https://pytorch-cn.readthedocs.io/zh/latest/package_references/functional/#_1)

## 测试

## 实现
