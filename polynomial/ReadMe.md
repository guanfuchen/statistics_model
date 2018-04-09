# 多项式拟合

- 模型
$f_M(x_i,w)=w_0+w_1 x_i+w_2 x_i^2+...+w_M x_i^M=\sum_{j=0}^{M}w_j x_i^j$
- 代价函数
$L(w)=\frac{1}{2}\sum_{i=0}^{n}(\sum_{j=0}^M{w_j x_i^j}-y_i)^2$
- 算法
代价函数对w求偏导数并赋值为0，求取的$\hat{w}$即为拟合的多项式系数。

---
## 证明

令$\mathbf{w}=(w_0,w_1,\cdots,w_M)^T$，$\mathbf{y}=(y_0,y_1,\cdots,y_n)^T$，$
X=\left[
\begin{matrix}
1& x_1& x_1^2& \cdots& x_1^M\\
1& x_2& x_2^2& \cdots& x_2^M\\
\vdots& \vdots& \vdots& \cdots& \vdots\\
1& x_n& x_n^2& \cdots& x_n^M
\end{matrix}
\right]
$，那么$f_M(x_i,w)=[X]_i^T\mathbf{w}$，$L(\mathbf{w})=\frac{1}{2}\sum_{i=0}^{n}(\sum_{j=0}^M{w_j x_i^j}-y_i)^2=\frac{1}{2}|X\mathbf{w}-\mathbf{y}|_2^2$，求得$\frac{\partial{L}}{\partial{\mathbf{w}}}=X^T(X\mathbf{w}-\mathbf{y})=0$，得到$\mathbf{w}=(X^T X)^{-1} X^T \mathbf{y}$


---
## 参考资料

[《统计学习方法》中关于求拟合多项式系数的问题？](https://www.zhihu.com/question/23483726)

[机器学习入门之多项式曲线拟合](https://blog.csdn.net/xwl198937/article/details/52210156)

[斯坦福机器学习课程 第三周 (4)正则化：解决过拟合问题](http://studyai.site/2016/09/04/%E6%96%AF%E5%9D%A6%E7%A6%8F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AF%BE%E7%A8%8B%20%E7%AC%AC%E4%B8%89%E5%91%A8%20(4)%E6%AD%A3%E5%88%99%E5%8C%96%EF%BC%9A%E8%A7%A3%E5%86%B3%E8%BF%87%E6%8B%9F%E5%90%88%E9%97%AE%E9%A2%98/)

[numpy.polyfit](https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.polyfit.html)

[李航《统计学习方法》多项式函数拟合问题V2](https://blog.csdn.net/xiaolewennofollow/article/details/46757657)
