# VAE

变分自动编码器。CVAE和CGAN表示基于类别Class不断生成模型，这两种结构互有优点：
- CVAE生成的图像很中规中矩，但是模糊。
- CGAN生成的图像清晰，但是喜欢乱来。

## KL散度

如果我们对于同一个随机变量x有两个单独的概率分布$P(x)$和$Q(x)$，我们可以使用KL散度来**衡量这两个分布的差异**。
$$D_{KL}(P||Q)=E_{x \sim P}[\log{\frac{P(x)}{Q(x)}}]$$
在离散型变量的情况下，KL散度衡量的是，当我们使用一种被设计成能够使得概率分布Q产生的消息的长度最小的编码。

### 一维高斯KL散度

假设有两个随机变量$x_1$和$x_2$，各自服从高斯分布$N_1(\mu_1, \sigma_1)$和$N_2(\mu_2, \sigma_2)$，计算它们的KL散度。

$$p_1(x)=\frac{1}{\sqrt{2 \pi \sigma_1^2}}e^{-\frac{(x-\mu_1)^2}{2 \sigma_1^2}}$$
$$p_2(x)=\frac{1}{\sqrt{2 \pi \sigma_2^2}}e^{-\frac{(x-\mu_2)^2}{2 \sigma_2^2}}$$
$$\int{p_1(x)}=1$$
$$\int{xp_1(x)}=\mu_1$$
$$\int{(x-\mu_1)p_1(x)}=\sigma_1^2$$
$$D_{KL}(p_1||p_2)=\int{p_1(x)\log{\frac{p_1(x)}{p_2(x)}}} dx$$
$$=\int{p_1(x)(\log{p_1(x)}-\log{p_2(x)})} dx$$
$$=\int{p_1(x)[\log{\frac{\sigma_2}{\sigma_1}}+\frac{(x-\mu_2)^2}{2 \sigma_2^2}-\frac{(x-\mu_1)^2}{2 \sigma_1^2}]} dx$$
$$=\log{\frac{\sigma_2}{\sigma_1}}+\int{p_1(x)[\frac{(x-\mu_2)^2}{2 \sigma_2^2}-\frac{(x-\mu_1)^2}{2 \sigma_1^2}]} dx$$
$$=\log{\frac{\sigma_2}{\sigma_1}}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac{1}{2}$$

当$N_2(\mu_2, \sigma_2)$为单位高斯分布时，即$\mu_2=0$，$\sigma_2=1$，$D_{KL}(p_1||p_2)=\log{\frac{1}{\sigma_1}}+\frac{\sigma_1^2+\mu_1^2}{2}-\frac{1}{2}$，当且仅当$\mu_1=0$，$\sigma_1=1$时，KL散度取得最小值。


## CVAE

CVAE是增加类别输入的VAE变种，输入为类别可以生成无限多该类别图像，架构如下图所示，其中E表示编码器，将输入$x$编码为高斯向量$z$，$G$表示生成器，将隐藏变量$z$生成为$x^{'}$，CVAE训练目标为$z$的分布尽可能接近$N(0,1)$的高斯分布，同时输出重构误差尽可能减小。
![CVAE架构](http://chenguanfuqq.gitee.io/tuquan2/img_2018_4/cvae_arch.png)
![CGAN架构](http://chenguanfuqq.gitee.io/tuquan2/img_2018_4/cgan_arch.png)
其中CVAE生成图像模糊的原因可以解释为该模型不容易找到合理的判断$x$和$x^{'}$的接近的标准，只能使用MSE来表示。
其中CGAN生成图像乱来的原因可以解释为其中鉴别器$D$的能力过小，生成器$G$对于某一个容易骗过鉴别器的模式不断重构的原因。
另一种综合这两个网络优点的是CVAE-GAN架构。
![CVAE-GAN架构](http://chenguanfuqq.gitee.io/tuquan2/img_2018_4/cvae_gan_arch.png)

## 相关资料

[Deep Learning Lecture 14: Karol Gregor on Variational Autoencoders and Image Generation](https://www.youtube.com/watch?v=P78QYjWh5sM) ML课程关于VAE和DRAW，课程较为浅显，不涉及太多过程。

[VAE（1）——从KL说起](https://zhuanlan.zhihu.com/p/22464760) 介绍了VAE，同时对KL散度进行了推导。

深度学习中文版第19章（近似推断和变分学习）

[深度神经网络生成模型：从 GAN VAE 到 CVAE-GAN](https://zhuanlan.zhihu.com/p/27966420) 对于VAE和GAN以及它们的结合CVAE-GAN都做了形象的解释。