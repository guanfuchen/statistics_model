# 线性回归模型

给定数据集$T={({\mathbf{x_1}}, y_1),({\mathbf{x_2}}, y_2),...,({\mathbf{x_m}}, y_m)}$，
其中${\mathbf{x_i}} \in R^{d \times 1}$，$y_i \in R$，$m$为样本数目，线性回归模型试图学得一个线性模型
以尽可能准确地预测实值输出标记。其中线性回归模型为$f(\mathbf{x})=w_1 x_1+w_2 x_2+...+w_d x_d + b$，向量
形式为$f(\mathbf{x})=\mathbf{w}^T \mathbf{x}+b$，其中$\mathbf{w} = (w_1 w_2 ... w_d)^T$，该模型由
参数$\mathbf{w}$和$b$确定。

这里首先考虑一维特征输入的情况，即$d=1$
损失函数选择均方误差：
$$(w^{*}, b^{*})=arg \min_{w.b} \sum_{i=1}^{m}(f(x_i)-y_i)^2$$
使用均方误差最小化来进行模型求解的方法称为“最小二乘法”，在线性回归中，最小二乘法就是试图找到一条直线。使所有样本到直线
欧式距离之和最小。
令$E(w,b)=\sum_{i=1}^{m}{(w x_i +b - y_i)}^2$，使用参数估计求$w$和$b$，
那么$\frac{\partial{E(w,b)}}{\partial{w}}=\sum_{i=1}^{m}{2 w x_i^2+2 x_i (b-y_i)}$，
$\frac{\partial{E(w,b)}}{\partial{b}}=\sum_{i=1}^{m}{2(w x_i - y_i) +2b}$，
所以令$\frac{\partial{E(w,b)}}{\partial{w}}=0$和$\frac{\partial{E(w,b)}}{\partial{b}}=0$，
可求得$w=\frac{\sum_{i=1}^{m}y_i (x_i-\overline{x})}{\sum_{i=1}^{m}x_i^2-\frac{1}{m}(\sum_{i=1}^{m}x_i^2)^2}$，
$b=\frac{1}{m}\sum{i=1}^{m}(y_i - w x_i)$，其中$\overline{x} = \frac{1}{m} \sum_{i=1}^{m} x_i$。

以上为一个特征的输入情况，接下来考虑一般情况，即$d>1$，那么$f(\mathbf{x_i})={\mathbf{w}^T}\mathbf{x}_i+b$，这称为多元线性回归。
为推导方便，这里令$\hat{\mathbf{w}}=(\mathbf{w}^T b)^T$，$\hat{\mathbf{x}}=(\mathbf{x}^T 1)^T$，则上述损失函数如下所示：
$E_{\hat{\mathbf{w}}}=(\mathbf{y}-\mathbf{\hat{x}}\mathbf{\hat{w}})^T (\mathbf{y}-\mathbf{\hat{x}}\mathbf{\hat{w}})$，
对$\hat{\mathbf{w}}$求导得到$\frac{\partial{E_{\hat{\mathbf{w}}}}}{\partial{\hat{\mathbf{w}}}}=2\mathbf{\hat{x}}^T(\mathbf{x\mathbf{\hat{w}}}-\mathbf{y})$，
令$\frac{\partial{E_{\hat{\mathbf{w}}}}}{\partial{\hat{\mathbf{w}}}}=0$，求得$\mathbf{\hat{w}^*}=\mathbf{(X^TX)^{-1}X^Ty}$，那么带入原模型方程即可。
然而，现实任务中$(X^TX)^{-1}$往往不是满秩矩阵。在许多任务中特征数将会大量超过样例数，导致$X$的列数多于行数，$X^TX$显然不满秩。此时可解出多个$\hat{w}$，它们都能使均方误差最小化。
选择哪一个解作为输出，将由学习算法的归纳偏好决定，常见的做法是引入正则化项。

## 参考：
- 李航《机器学习》，第三章中的线性模型中的线性回归
