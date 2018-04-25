# GAN

生成对抗网络，是最近非常流行的一种非监督学习网络，能够从少量的样本中学习到与监督学习相当的知识。

> GAN优势很多：根据实际的结果，看上去产生了更好的样本；GAN能训练任何一种生成器网络；GAN不需要设计遵循任何种类的因式分解的模型，任何生成器网络和任何鉴别器都会有用；GAN无需利用马尔科夫链反复采样，无需在学习过程中进行推断，回避了近似计算棘手的概率的难题。

> GAN主要存在的以下问题：网络难以收敛，目前所有的理论都认为GAN应该在纳什均衡上有很好的表现，但梯度下降只有在凸函数的情况下才能保证实现纳什均衡。


---
## 参考资料

[Generative Models](https://blog.openai.com/generative-models/#vae) OpenAI介绍相关生成模型，包括VAE、GAN和自回归模型（如PixelRNN）。

[dcgan_code](https://github.com/Newmu/dcgan_code) Deep Convolutional Generative Adversarial Networks深度卷积生成对抗网络。

[improved-gan](https://github.com/openai/improved-gan) 论文Improved Techniques for Training GANs代码，提供了提升GAN网络训练tricks。

[GAN-based Segmentation](http://blog.leanote.com/post/sunalbert/GAN-based-Segmentation) GAN和语义分割结合的分割策略。

[GAN论文整理](https://www.jianshu.com/p/2acb804dd811) 整理了GAN方面相关的论文。

[ICLR 2017 | GAN Missing Modes 和 GAN](https://mp.weixin.qq.com/s?__biz=MzAwMjM3MTc5OA==&mid=2652692183&idx=1&sn=b436cba6a6fcd19dccaddccd42cb0f11) 介绍了ICLR2017上一些GAN网络相关的论文。

[生成式对抗网络GAN有哪些最新的发展，可以实际应用到哪些场景中？](https://www.zhihu.com/question/52602529) 知乎上关于生成对抗网络的讨论。

[GAN Lecture 1: Introduction of Generative Adversarial Network (GAN)](https://www.youtube.com/watch?v=G0dZc-8yIjE) 李宏毅关于GAN的课程。

[dcgan main.py](https://github.com/pytorch/examples/blob/master/dcgan/main.py) deep convolution gan深度卷积GAN pytorch代码。下载lsun bedroom数据集进行实现。

[Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch)](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f) pytorch实现简单的GAN网络。[pytorch-generative-adversarial-networks代码](https://github.com/devnag/pytorch-generative-adversarial-networks)

