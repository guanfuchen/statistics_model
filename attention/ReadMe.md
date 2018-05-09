# Attention

这里主要介绍Attention机制在图像标注上的应用，首先介绍未引入Attention机制的图像标注系统。

---
## 图像标注系统

### 数据集
数据集使用coco，其中提供了python API供使用，[cocoapi](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI)。

---
## 参考资料

[Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

[理解LSTM/RNN中的Attention机制](http://www.jeyzhang.com/understand-attention-in-rnn.html)

[ATTENTION MECHANISM](https://blog.heuritech.com/2016/01/20/attention-mechanism/) 介绍了attention机制在图像标注中的内部结构。对应的中文翻译[Attention Model 注意力机制](https://ilewseu.github.io/2018/02/12/Attention%20Model%20%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/)

[image_captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning) 使用LSTM进行图像标注（未使用Attention机制）。

[image-captioning](https://github.com/surgicaI/image-captioning) 该仓库大体相同，增加Attention机制同时增加VGG，ResNet和LSTM，GRU等模块的比较。

[Image / Video Captioning](https://handong1587.github.io/deep_learning/2015/10/09/captioning.html#blogs) handong1587跟踪的图像标注相关文章。
