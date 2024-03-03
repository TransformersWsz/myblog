---
title: ViT的若干细节
mathjax: true
toc: true
date: 2024-03-03 16:01:20
updated: 2024-03-03 16:01:20
categories:
- Machine Learning
tags:
- VIT
- CNN
---
之前只看了ViT的大概结构，具体的模型细节和代码实现知之甚少。随着ViT逐渐成为CV领域的backbone，有必要重新审视下。

<!--more-->

![ViT](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.ltb9u1l3.webp)

## 输入
为了将图片处理成序列格式，很自然地想到将图片分割成一个个patch，再把patch处理成token。

![patch](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.54xi0lkn9a.webp)

假设图片大小为 $224 \times 224 \times 3$ (即 $H \times W \times C$ )，每个patch大小为 $16 \times 16 \times 3$，那么序列长度就是 $196$，序列的形状是 $196 \times 768$。

如何将大小为 $16 \times 16 \times 3$ 的patch，映射为 $768$ 维的token？源码是直接将其[reshape](https://github.com/lucidrains/vit-pytorch/blob/5578ac472faf3903d4739ba783f3875b77177e57/vit_pytorch/vit.py#L96)

{% note danger %}
在reshape之后，还需要过一层$768 \times 768$的embedding层。因为reshape后的$768$维向量是参数无关的，不参与梯度更新，过完embedding层，即拥有了token embedding的语义。
{% endnote %}

#### 处理成patch的好处
- 减少计算量：如果按照pixel维度计算self-attention，那复杂度大大增加。patch size越大，复杂度越低。stable diffusion也是这个思路，在latent space进行扩散，而不是pixel
- 减少图像冗余信息：图像是有大量冗余信息的，处理成patch不影响图片语义信息


## 参考
- [再读VIT，还有多少细节是你不知道的](https://mp.weixin.qq.com/s/kcqYF-Z3AwbPLQUOozyI0Q)