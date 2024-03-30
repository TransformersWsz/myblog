---
title: 大模型融合方法-DARE
mathjax: true
toc: true
date: 2024-03-30 23:46:51
updated: 2024-03-30 23:46:51
categories:
- NLP
tags:
- LLM
- 模型融合
---
LLM在SFT之后会产生大量的冗余参数(delta参数)，阿里团队提出DARE方法来消除delta参数，并将其合并到PRE模型中，从而实现多源模型能力的吸收。


DARE无需GPU重新训练，其思路非常简单，就跟dropout类似：
$$
\begin{gathered}
\boldsymbol{m}^t \sim \operatorname{Bernoulli}(p) \\
\widetilde{\boldsymbol{\delta}}^t=\left(\mathbf{1}-\boldsymbol{m}^t\right) \odot \boldsymbol{\delta}^t \\
\hat{\boldsymbol{\delta}}^t=\widetilde{\boldsymbol{\delta}}^t /(1-p)  \\
\boldsymbol{\theta}_{\mathrm{DARE}}^t=\hat{\boldsymbol{\delta}}^t+\boldsymbol{\theta}_{\mathrm{PRE}}
\end{gathered}
$$
两个步骤：
1. drop：随机mask参数为0
2. rescale：对保存的参数rescale，这样可以保证神经元期望值不变：$E_{not_{mask}}=x,E_{mask}=\frac{p*x}{p}$

传统的模型融合只是对神经元进行加权求和，这样会导致模型能力骤降。DARE方法通过dropout避免了这种问题。

## 多源模型融合
$$
\begin{gathered}
\boldsymbol{\theta}_{\mathrm{DARE}}^{t_k}=\operatorname{DARE}\left(\boldsymbol{\theta}_{\mathrm{SFT}}^{t_k}, \boldsymbol{\theta}_{\mathrm{PRE}}\right), \text { for } 1 \leq k \leq K, \\
\boldsymbol{\theta}_{\mathrm{M}}=\boldsymbol{\theta}_{\mathrm{PRE}}+\lambda \cdot \sum_{k=1}^K \hat{\boldsymbol{\delta}}^{t_k}=\boldsymbol{\theta}_{\mathrm{PRE}}+\lambda \cdot \sum_{k=1}^K\left(\boldsymbol{\theta}_{\mathrm{DARE}}^{t_k}-\boldsymbol{\theta}_{\mathrm{PRE}}\right) .
\end{gathered}
$$

流程图：

![procedure](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.3d4k8ni4bl.png)



## 实验结果
![result](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.5q76puxcia.webp)

___

## 参考
- [丢弃99%的参数！阿里团队提出语言模型合体术，性能暴涨且无需重新训练和GPU](https://mp.weixin.qq.com/s/YiqWovBUXIbzmUbL6uT-8g)
- [MergeLM](https://github.com/yule-BUAA/MergeLM)