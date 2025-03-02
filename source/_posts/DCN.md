---
title: DCN
mathjax: true
toc: true
date: 2025-03-02 23:13:38
updated: 2025-03-02 23:13:38
categories:
- 搜广推
tags:
- Cross-Features
---
DCN是DeepFM的升级版，后者是只能做二阶交叉特征，随着阶数上升，模型复杂度大幅提高，且FM网络层较浅，表达能力有限。google团队通过构建深度交叉网络来自动进行特征的高阶交叉，且时空复杂度均为线性增长，极大提升了模型性能。

<!--more-->

## 模型结构
整体网络结构跟DeepFM类似：

![model](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.9kgbmcdyez.webp)

特征交叉细节：

![cross](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.8oju6w5gkd.webp)

## 代码实现

代码其实非常简单：

```python
def cross_net(self, inputs):
    # 进行特征交叉时的x_0一直没有变，变的是x_l和每一层的权重
    x_0 = inputs # B x dims 
    x_l = x_0
    for i in range(self.layer_nums):
        # 将x_l的第一个维度与w[i]的第0个维度计算点积
        xl_w = tf.tensordot(x_l, self.W[i], axes=(1, 0)) # B, 
        xl_w = tf.expand_dims(xl_w, axis=-1) # 在最后一个维度上添加一个维度 # B x 1
        cross = tf.multiply(x_0, xl_w) # B x dims
        x_l = cross + self.b[i] + x_l
    return x_l
```
这里的 `cross` 其实是相当于学习残差。

## 实验结果
就随便看看吧，baselines提到了FM、LR，但只字不提跟它们的性能比较，无语。。。（Wide&Deep依赖于大量人工先验来选择交叉特征，DCN只跟自动交叉特征的方法比，例如FM等）

![logloss](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.lvve45ssr.webp)

![parameter](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.64dzu9ezcs.webp)

___

## 参考
- [DCN.md](https://github.com/datawhalechina/fun-rec/blob/master/docs/ch02/ch2.2/ch2.2.2/DCN.md)
- [Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754)