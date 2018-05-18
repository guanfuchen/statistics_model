# LDA
Linear Discriminant Analysis，中文线性判别分析，简称为LDA。其中LDA和PCA都是常用的降维技术。PCA主要是从特征的协方差角度，去找到比较好的投影方式（压缩方式）。LDA更多考虑了标注，即希望投影后的不同类别之间的数据点的距离更大，同一类别的数据点更紧凑。

浅显来看，LDA方法考虑的是，对于一个多类别的分类问题，想要把它映射到一个低维空间中，如一维空间从而达到降维的目的，我们希望映射后的数据间，不同类别之间距离越远，同一类别之间距离越近，这样两个类别就比较好区分。

因此LDA方法分别计算“within-class”的分散程度$S_w$和“between-class”的分散程度$S_b$，同时希望$S_b/S_w$越大越好，从而找到最合适的映射向量。

---
## 参考资料

[如何理解LDA算法？能够简洁明了地说明一下LDA算法的中心思想吗？](https://www.zhihu.com/question/34305879)

[A Tutorial on Data Reduction](http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_LDA09.pdf) 主要参考这个教程，读完非常清楚。

[线性判别分析LDA原理总结](https://www.cnblogs.com/pinard/p/6244265.html) 依然是刘建平大神的LDA算法总结。

[机器学习降维算法二：LDA（Linear Discriminant Analysis）](http://www.cnblogs.com/xbinworld/archive/2011/11/24/lda.html)

[线性判别分析（Linear Discriminant Analysis）（一）](http://www.cnblogs.com/jerrylead/archive/2011/04/21/2024384.html) 这篇推导较为仔细。

Face Recognition with Python 这篇文章介绍了PCA和LDA在人脸识别上的应用，主要是EigenFaces和FisherFaces，其中也包括了算法核心推导，值得阅读。
