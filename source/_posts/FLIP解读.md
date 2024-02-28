---
title: FLIP解读
mathjax: true
toc: true
date: 2024-02-06 17:22:20
categories:
- Machine Learning
tags:
- CLIP
- MAE
- Masked Autoencoders
- Contrastive Learning
---
FLIP由CLIP改进而来，其思想非常简单，通过在图片侧mask掉相当比例的patch（无须重构patch），实现速度和准确性的双重提升。

<!--more-->

## 模型结构

![model](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.k290d3y8ylc.webp)

受MAE启发，FLIP对图像进行了mask来预训练。该方法有两方面收益：
- 速度：ViT对图像编码的计算量大幅减少，训练速度更快
- 准确性：相同的显存可以存放更多的batch，从而构造更多的图文对进行对比学习，准确性得以提高

![speed](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.2vrg1obvy7i0.png)

值得注意的是，该预训练任务没有重构patch，个人理解：
- 图片本身就包含了大量的冗余信息，mask掉部分patch不影响图片理解
- mask部分patch，可以强制两边编码器去学习对方的上下文语义信息

## 实验结果

FLIP在下游实验的结果一片绿：

![experiment](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.2vmbpzmm3540.webp)

#### 消融实验
![ablation](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.751rg20p7vg0.webp)
- 作者尝试在图像上的不同mask比例，50%最佳
- 作者也尝试了在文本上做mask，但性能略微有所下降
- 重构patch没有收益

___

## 参考
- [简单高效！何恺明大神之多模态预训练FLIP](https://mp.weixin.qq.com/s/e2OdCN6jMK-kWapaEaoO1A)