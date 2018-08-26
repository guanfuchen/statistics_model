# learning rates

该章节主要介绍常用的learning rates和schedules方法。

---
## 相关论文
- Cyclical Learning Rates for Training Neural Networks，该篇论文针对原始地针对单个学习率随着训练时间单调递减的思路，提出了一种全局的周期性的学习率调整思路。
- WNGrad: Learn the Learning Rate in Gradient Descent
- Don't Decay the Learning Rate, Increase the Batch Size
- On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima
- Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
- Systematic evaluation of CNN advances on the ImageNet
- The Effects of Hyperparameters on SGD Training of Neural Networks，对SGD训练神经网络调整超参数的影响总结。
- Optimal Distributed Online Prediction using Mini-Batches
- Efficient Mini-batch Training for Stochastic Optimization
- Optimization Methods for Large-Scale Machine Learning
- Adaptive stepsizes for recursive estimation with applications in approximate dynamic programming，自适应学习率reivew，可以参考。
- Adaptive subgradient methods for online learning and stochastic optimization AdaGrad自适应梯度下降方法。
- An overview of gradient descent optimization algorithms 梯度下降优化算法整体概述。


---
## 参考资料
- [深度学习炼丹师的养成之路之——Batch size/Epoch/Learning Rate的设置和学习策略](https://blog.csdn.net/qiusuoxiaozi/article/details/78456544) 该博客介绍了batch size、epoch和学习率的设置对模型训练效果的影响，同时这些方法也针对不同的优化器方法不同，比如Adam优化器针对不同的数据有很强的自适应性。
- [如何理解深度学习分布式训练中的large batch size与learning rate的关系？](https://www.zhihu.com/question/64134994) 知乎上关于lr有相关的较好的论文资料可以参考。
- [系统学习深度学习（二十五）--CNN调优总结](https://blog.csdn.net/App_12062011/article/details/64439627)
