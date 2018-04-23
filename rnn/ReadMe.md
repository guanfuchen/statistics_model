# rnn

循环神经网络

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_4/rnn_simple_2.png)

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_4/rnn_simple_1.png)

---
## 前向传播

- 隐藏和输入concat
$$z_t = [h_{t-1}, x_t]$$
- 下一时刻隐藏
$$h_t = w_h \cdot h_{t-1}$$
- 下一时刻输出$v_t$
$$v_t = w_v \cdot z_{t}$$
- 下一时刻输出$y_t$
$$y_t = softmax(v_t)$$

---
## 反向传播

---
# lstm

长短期记忆网络

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_4/rnn_forward.png)

---
## 前向传播

- 隐藏和输入concat
$$z_t = [h_{t-1}, x_t]$$
- 遗忘门
$$f_t=\sigma(W_f \cdot z_t+b_f)$$
- 输入门
$$i_t=\sigma(W_i \cdot z_t+b_i)$$
- $\bar{C}$
$$\bar{C}_t=\tanh(W_C \cdot z_t+b_C)$$
- C
$$C_t=f_t * C_{t-1}+i_t * \bar{C}_t$$
- 输出门
$$o_t=\sigma(W_o \cdot z_t+b_o)$$
- h
$$h_t=o_t * \tanh(C_t)$$
- v
$$v_t=W_v \cdot h_t+b_v$$
- $\bar{y}$
$$\bar{y_t}=softmax(v)$$
- Loss
$$L(y,\bar{y}) = -\sum_{t=1}^{\tau}{\sum_{j}{y_{tj}\log{\bar{y}_{tj}}}}$$

---
## 反向传播


---
## 存在的问题



### 梯度

计算$t=\tau$时刻的$\frac{\partial{L}}{\partial{C_{\tau}}}$和$\frac{\partial{L}}{\partial{h_{\tau}}}$，然后通过递推公式计算$\frac{\partial{L}}{\partial{C_{t}}}$和$\frac{\partial{L}}{\partial{h_{t}}}$。
$$\frac{\partial{L}}{\partial{h_{\tau}}}=W_v^T \cdot (y_{\tau}-\bar{y_{\tau}})$$

$$\frac{\partial{L}}{\partial{C_{\tau}}}=\frac{\partial{L}}{\partial{h_{\tau}}}*o_{\tau}*(1-\tanh^2{C_{\tau}})$$

前一时刻和后一时刻之间的递推关系：

$$\frac{\partial{L}}{\partial{h_{t}}}=W_v^T \cdot (y_{t}-\bar{y}_{t})$$

$$\frac{\partial{L}}{\partial{C_{t}}}=\frac{\partial{L}}{\partial{h_{t}}}*o_{t}*(1-\tanh^2{C_{t}})+f_{t+1}*\frac{\partial{L}}{\partial{C_{t+1}}}$$

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_4/rnn_forward_size.png)

### 模型梯度
$$\mathrm{d}W_v=\sum_{t=1}^{\tau}(y_t-\bar{y_t}) \cdot h_t^T$$
$$\mathrm{d}b_v=\sum_{t=1}^{\tau}(y_t-\bar{y_t})$$
$$\mathrm{d}W_o=\sum_{t=1}^{\tau}\frac{\partial{L}}{\partial{h_{t}}}*\tanh{C_t}*o_t*(1-o_t) \cdot z_t^T$$
$$\mathrm{d}b_o=\sum_{t=1}^{\tau}\frac{\partial{L}}{\partial{h_{t}}}*\tanh{C_t}*o_t*(1-o_t)$$
$$\mathrm{d}W_f=\sum_{t=1}^{\tau}\frac{\partial{L}}{\partial{C_{t}}}*C_{t-1}*f_t*(1-f_t) \cdot z_t^T$$
$$\mathrm{d}b_f=\sum_{t=1}^{\tau}\frac{\partial{L}}{\partial{C_{t}}}*C_{t-1}*f_t*(1-f_t)$$
$$\mathrm{d}W_C=\sum_{t=1}^{\tau}\frac{\partial{L}}{\partial{C_{t}}}*i_t*(1-\tanh^2(\bar{C_t})) \cdot z_t^T$$
$$\mathrm{d}b_C=\sum_{t=1}^{\tau}\frac{\partial{L}}{\partial{C_{t}}}*i_t*(1-\tanh^2(\bar{C_t}))$$
$$\mathrm{d}W_i=\sum_{t=1}^{\tau}\frac{\partial{L}}{\partial{C_{t}}}*i_t*(1-i_t)*\bar{C_t} \cdot z^T$$
$$\mathrm{d}b_i=\sum_{t=1}^{\tau}\frac{\partial{L}}{\partial{C_{t}}}*i_t*(1-i_t)*\bar{C_t}$$

其中符号$ \cdot $表示H乘积，$*$表示矩阵乘法。

$$(x_1, x_2) \cdot (y_1, y_2) = (x_1*x_2, y_1*y_2)$$

---
# 参考资料

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/index.html) cs231n中RNN这一章节。

[CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/index.html) cs224d中RNN这一章节。

[LSTM模型与前向反向传播算法](http://www.cnblogs.com/pinard/p/6519110.html) 刘建平系列博客，讲解非常清晰。

[循环神经网络(RNN)模型与前向反向传播算法](http://www.cnblogs.com/pinard/p/6509630.html) 刘建平系列博客。

[Vanilla LSTM with numpy](http://blog.varunajayasiri.com/numpy_lstm.html) 使用numpy实现了LSTM。

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[理解 LSTM 网络](https://www.yunaitong.cn/understanding-lstm-networks.html) understanding lstm的中文版本。

[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) understanding LSTM的博主写的一篇关于NLP中Word Embeddings的相关衍生。

[深度学习，自然语言处理，及表征方法 – ‘Deep Learning, NLP, and Representations’](https://cindyxiaoxiaoli.wordpress.com/2014/10/22/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%EF%BC%8C%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%EF%BC%8C%E5%8F%8A%E8%A1%A8%E5%BE%81%E6%96%B9%E6%B3%95-deep-learning-nlp-and-representations/) Deep Learning, NLP, and Representations中文翻译。

[Practical PyTorch: Classifying Names with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/char-rnn-classification.ipynb) 使用pytorch构建RNN模型通过姓名分类国籍。

[Classifying Names with a Character-Level RNN](http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) pytorch RNN实践资料。

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) Andrej Karpathy大神char-rcnn的博客，想象力丰富。[min-char-rnn.py代码](https://gist.github.com/karpathy/d4dee566867f8291f086)

[循环神经网络惊人的有效性（上）](https://zhuanlan.zhihu.com/p/22107715) The Unreasonable Effectiveness of Recurrent Neural Networks中文翻译。

[rnn-from-scratch](https://github.com/pangolulu/rnn-from-scratch) 使用numpy实现rnn。

[翻译WILDML RNN系列教程 第一部分 RNN简介](http://friskit.me/2016/10/09/translation-wildml-recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) 关于RNN的系列教程，值得参考。[rnn-tutorial-rnnlm实现代码](https://github.com/dennybritz/rnn-tutorial-rnnlm)

[RNN(循环神经网络)推导](http://manutdzou.github.io/2016/07/11/RNN-backpropagation.html)

[CS224d－Day 5: RNN快速入门](http://machinelearninghandbook.com/2018/01/cs224d%EF%BC%8Dday-5-rnn%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8/) 简单介绍。

[rnn.py pytorch](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py)

[循环神经网络RNN 梯度推导(BPTT)](https://ilewseu.github.io/2017/12/30/RNN%E7%AE%80%E5%8D%95%E6%8E%A8%E5%AF%BC/) 推导过程较为清晰。

[第7课 循环神经网络与LSTM（寒）](https://www.youtube.com/watch?v=nLPTyL4aLzM) 七月算法深度学习课程，课程内容基本上是网上博客相关内容，可以作为复习。

[循环神经网络和递归神经网络1](https://www.youtube.com/watch?v=OUU4o6WOai8) 深度学习这本书RNN这一章节的视频总结。

[LSTM: A Search Space Odyssey](https://arxiv.org/pdf/1503.04069.pdf) LSTM各种变体对比。[lstm-odyssey实现代码](https://github.com/fomorians/lstm-odyssey) [An LSTM Odyssey 作者博客](https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93)

[An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf) 多种RNN和LSTM比较。

[awesome-rnn](https://github.com/kjw0612/awesome-rnn)

Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling 论文[librnn实现代码](https://github.com/jych/librnn)

[循环和递归网络学习笔记](https://zxth93.github.io/2017/09/16/%E5%BE%AA%E7%8E%AF%E5%92%8C%E9%80%92%E5%BD%92%E7%BD%91%E7%BB%9C%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/index.html) 作者阅读deep learning这本书总结的RNN相关知识。

