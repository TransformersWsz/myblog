---
title: DANN & GRL
mathjax: true
toc: true
date: 2024-09-24 02:46:43
updated: 2024-09-24 02:46:43
categories:
- Machine Learning
tags:
- Gradient Reversal
- Domain Adaptation
---

域自适应是指在目标域与源域的数据分布不同但任务相同下的迁移学习，从而将模型在源域上的良好性能迁移到目标域上，极大地缓解目标域标签缺失严重导致模型性能受损的问题。

介绍一篇经典工作 [DANN](https://proceedings.mlr.press/v37/ganin15.pdf) ：

<!--more-->

## 模型结构

![model](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.8ojnugoi4m.webp)

在训练阶段需要预测如下两个任务：
- 实现源域数据集准确分类，即图像分类误差的最小化，这与正常分类任务保持一致
- 实现源域和目标域准确分类，即域分类器的误差最小化。而特征提取器的目标是最大化域分类误差，使得域分类器无法分辨数据是来自源域还是目标域，从而让特征提取器学习到域不变特征(domain-invariant)。也就是说特征提取器和域分类器的目标是相反的
  - 本质上就是让特征提取器不要过拟合源域，要学习出源域和目标域的泛化特征
  - 这两个网络对抗训练，DANN通过GRL层使特征提取器更新的梯度与域判别器的梯度相反，构造出了类似于GAN的对抗损失，又通过该层避免了GAN的两阶段训练过程，提升模型训练稳定性

## GRL
GRL是作用在特征提取器上的，对其参数梯度取反，具体实现如下：
```python
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
```

___

## 参考
- [【深度域自适应】一、DANN与梯度反转层（GRL）详解](https://zhuanlan.zhihu.com/p/109051269)