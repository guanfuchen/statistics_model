# 矩阵求导

机器学习中存在大量矩阵、向量（行、列向量）、标量相互之间的求导运算。求导规则根据不同的
布局结果不同，矩阵求导总共有两种布局，分子布局和分母布局，也就是结果按照分子的结构还是
分母的结构。例如向量$\mathbf{y}$对标量$x$求导，其中$\mathbf(y)$等向量都是列向量。
$$\mathbf{y}=
\left[
\begin{matrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{matrix}
\right]$$
分子布局下：
$$\frac{\partial{\mathbf{y}}}{{\partial{x}}}=
\left[
\begin{matrix}
\frac{\partial{y_1}}{{\partial{x}}} \\
\frac{\partial{y_2}}{{\partial{x}}} \\
\vdots \\
\frac{\partial{y_m}}{{\partial{x}}} \\
\end{matrix}
\right]$$
分母布局下：
$$\frac{\partial{\mathbf{y}}}{{\partial{x}}}=
\left[
\begin{matrix}
\frac{\partial{y_1}}{{\partial{x}}}&
\frac{\partial{y_2}}{{\partial{x}}}&
\cdots&
\frac{\partial{y_m}}{{\partial{x}}}
\end{matrix}
\right]$$
可见，分子布局和分母布局互为装置关系，整个推导都需要遵循统一布局，分子布局或者分母布局。

这里推导标量$y$对向量$\mathbf{x}$求导（下面的推导都是基于分母布局）：
$$\frac{\partial{y}}{\partial{\mathbf{x}}}=
\left[
\begin{matrix}
\frac{\partial{y}}{{\partial{x_1}}} \\
\frac{\partial{y}}{{\partial{x_2}}} \\
\vdots \\
\frac{\partial{y}}{{\partial{x_m}}}
\end{matrix}
\right]
$$
标量对向量求导和向量对标量求导刚好反过来

向量$\mathbf{y}$对向量$\mathbf{x}$求导：
$$\mathbf{y}=
\left[
\begin{matrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{matrix}
\right]$$

$$\mathbf{x}=
\left[
\begin{matrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{matrix}
\right]$$

$$\frac{\partial{\mathbf{y}}}{\partial{\mathbf{x}}}=
\left[
\begin{matrix}
\frac{\partial{\mathbf{y}}}{{\partial{x_1}}} \\
\frac{\partial{\mathbf{y}}}{{\partial{x_2}}} \\
\vdots \\
\frac{\partial{\mathbf{y}}}{{\partial{x_n}}}
\end{matrix}
\right]=
\left[
\begin{matrix}
\frac{\partial{y_1}}{{\partial{x_1}}}& \frac{\partial{y_2}}{{\partial{x_1}}}& \cdots & \frac{\partial{y_m}}{{\partial{x_1}}}\\
\vdots& \vdots& \cdots\\
\frac{\partial{y_1}}{{\partial{x_n}}}& \frac{\partial{y_2}}{{\partial{x_n}}}& \cdots & \frac{\partial{y_m}}{{\partial{x_n}}}
\end{matrix}
\right]
$$

标量对矩阵求导：
$$\frac{\partial{y}}{\partial{\mathbf{X}}}=
\left[
\begin{matrix}
\frac{\partial{y}}{{\partial{X_{11}}}}& \frac{\partial{y}}{{\partial{X_{12}}}}& \cdots & \frac{\partial{y}}{{\partial{X_{1q}}}}\\
\vdots& \vdots& \cdots\\
\frac{\partial{y}}{{\partial{X_{p1}}}}& \frac{\partial{y}}{{\partial{X_{p2}}}}& \cdots & \frac{\partial{y}}{{\partial{X_{p1}}}}
\end{matrix}
\right]
$$

矩阵对标量求导：
$$\frac{\partial{\mathbf{X}}}{\partial{y}}=
\left[
\begin{matrix}
\frac{{\partial{\mathbf{X_{1}}}}}{\partial{y}}& \frac{{\partial{\mathbf{X_{2}}}}}{\partial{y}}& \cdots & \frac{{\partial{\mathbf{X_{p}}}}}{\partial{y}}
\end{matrix}
\right]=
\left[
\begin{matrix}
\frac{{\partial{{X_{11}}}}}{\partial{y}}& \frac{{\partial{{X_{21}}}}}{\partial{y}}& \cdots & \frac{{\partial{{X_{p1}}}}}{\partial{y}} \\
\vdots& \vdots& \cdots& \vdots \\
\frac{{\partial{{X_{1q}}}}}{\partial{y}}& \frac{{\partial{{X_{2q}}}}}{\partial{y}}& \cdots & \frac{{\partial{{X_{pq}}}}}{\partial{y}}
\end{matrix}
\right]
$$

总结，在分母布局下，行向量对标量求导是列向量，列向量对行向量求导是行向量，而在分子布局下，由于分子是行向量，
那么结果根据分子也是行向量，同理列向量也是如此，矩阵可以分解为行向量的列向量，或者列向量的行向量均可。
标量对行向量求导是行向量，对列向量求导也是列向量，对矩阵求导也是相同，即，在分母规则下，凡是对标量求导，结果的形式
都要装置，而标量对向量和矩阵求导则位置则保持不懂。

维度分析
利用分母布局，向量、矩阵之间的求导法则可以获取维度的变化。
$$\frac{\partial{\mathbf{A}\mathbf{x}}}{\partial{\mathbf{x}}}$$
分析如下，结果是$\mathbf{A}$或者
$\mathbf{A}^T$，首先，在分母布局下，$A \in R^{m \times n}$，$x \in R^{n \times 1}$，
那么$\mathbf{A}x \in R^{m \times 1}$，那么$\frac{\partial{\mathbf{A}\mathbf{x}}}{\partial{\mathbf{x}}} \in R^{n \times m}$，结果就是$\mathbf{A}^T$

$$\frac{\partial{\mathbf{A}\mathbf{u}}}{\partial{\mathbf{x}}}=\frac{\partial{\mathbf{u}}}{\partial{\mathbf{x}}}\mathbf{A}^T$$

同理$A \in R^{m \times n}$，$u \in R^{n \times 1}$，$x \in R^{p \times 1}$，
结果$\frac{\partial{\mathbf{A}\mathbf{u}}}{\partial{\mathbf{x}}} \in R^{p \times m}$，
另外$\frac{\partial{\mathbf{u}}}{\partial{\mathbf{x}}} \in R^{p \times n}$，那么应该右乘$\mathbf{A}^T \in R^{n \times m}$

下面这些式子都可以使用相同的方法证明分析：
$$\frac{\partial{a\mathbf{u}}}{\mathbf{x}}=a\frac{\partial{\mathbf{u}}}{\mathbf{x}}+\frac{\partial{a}}{\mathbf{x}}{\mathbf{u}}^T$$
$$\frac{\partial{\mathbf{x}^T \mathbf{A}} \mathbf{y}}{\mathbf{\partial{x}}}=\frac{\partial{\mathbf{y}}}{\mathbf{\partial{x}}}\mathbf{A}^T \mathbf{x} + \mathbf{A} \mathbf{y}$$


[闲话矩阵求导](http://xuehy.github.io/blog/2014/04/18/2014-04-18-matrixcalc/index.html)

[Matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus)

[矩阵求导术（上）](https://zhuanlan.zhihu.com/p/24709748)

[矩阵求导问题](https://saicoco.github.io/matrix/)