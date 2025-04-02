---
title: COBRA详解
mathjax: true
toc: true
date: 2025-04-03 00:24:38
updated: 2025-04-03 00:24:38
categories: 
- 搜广推
tags:
- Generative Recommendation
- LLM
---
这是一篇生成式推荐用于召回场景的工作，其建模范式仍旧是输入端根据用户行为序列构造prompt，输出端预测next item。该工作巧妙地将稀疏ID与稠密向量表征级联融合起来，达到了SOTA水平。

<!--more-->

## 传统方法对比

| 方案类型          | 核心技术                     | 局限性                     |
|-------------------|----------------------------|---------------------------|
| 纯文本+LLM        | 直接使用广告文本特征        | 输入过长，资源消耗大        |
| 短语表征          | 关键词压缩表达              | 信息丢失严重               |
| 稠密表征+对比学习  | 端到端向量编码              | 建模复杂度高，缺少兴趣探索  |
| 稀疏ID生成        | RQ-VAE量化技术             | 信息损失导致细粒度捕捉弱    |


## COBRA介绍
**稀疏ID可以唯一表示item，有很好的区分性，但丧失了对item的细粒度信息刻画。纯文本可以准确可以item属性，但构造成prompt太长，套入到LLM
中会导致资源消耗过大**。那么如何结合两者的优点呢？


COBRA首先根据codebook生成item的稀疏ID，**该ID可以理解为item的大类别。既不过于精细，像unique id，又不过于宽泛**。然后将ID序列输入到Transformer Decoder中预测稠密向量。

![model](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.wiqfmnvsa.webp)

#### 离线训练

两个预测任务的损失函数如下：
$$
\mathcal{L}_{\text {sparse }}=-\sum_{t=1}^{T-1} \log \left(\frac{\exp \left(z_{t+1}^{I D_{t+1}}\right)}{\sum_{j=1}^C \exp \left(z_{t+1}^j\right)}\right) \\

\left.\left.\mathcal{L}_{\text {dense }}=-\sum_{t=1}^{T-1} \log \frac{\exp \left(\cos \left(\hat{\mathbf{v}}_{t+1} \cdot \mathbf{v}_{t+1}\right)\right)}{\sum_{\text {item }_j \in \text { Batch }} \exp \left(\operatorname { c o s } \left(\hat{\mathbf{v}}_{t+1}, \mathbf{v}_{\text {item }}^j\right.\right.} \mathbf{}\right)\right)
$$

ID预测就是经典的多分类任务，dense vector就是经典的对比学习任务。

#### 在线推理

稀疏ID生成：