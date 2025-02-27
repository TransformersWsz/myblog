---
title: PEPNet
mathjax: true
toc: true
date: 2025-02-28 01:20:01
updated: 2025-02-28 01:20:01
categories:
- 搜广推
tags:
- Multi-Task
---
鉴于PEPNet已经是多场景、多任务建模的baseline，这里有必要详细讲解一下。

<!--more-->

## 背景动机

![example](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.5xarunoecl.webp)

用户在多场景、多任务下的行为存在共性和差异性，如何用联合建模来捕捉这些特性又避免跷跷板效应成为一大难点。


## 模型结构

![model](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.7egwweyjxj.webp)

模型分为三部分：

#### Gate Neural Unit(Gate NU)

![gate](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.3gojfqshav.webp)

PEPNet的核心组件，由两层MLP构成，利用门控机制动态激活模型参数：
$$
\begin{aligned}
\mathbf{x}^{\prime} &= Relu(xW+b) \\
\boldsymbol{\delta} &= \gamma * \operatorname{Sigmoid}\left(\mathbf{x}^{\prime} \mathbf{W}^{\prime}+\boldsymbol{b}^{\prime}\right), \boldsymbol{\delta} \in[0, \gamma]
\end{aligned}
$$

#### Embedding Personalized Network(EPNet)

在特征embedding层，显示注入场景的先验信息，从而强化场景个性化特征（可以理解为与该场景相关的特征筛选）：
$$
\begin{aligned}
E &= E\left(\mathcal{F}_S\right) \oplus E\left(\mathcal{F}_D\right) \\
\delta_{\text {domain }} &= \mho_{e p}\left(E\left(\mathcal{F}_d\right) \oplus(\oslash(\mathbf{E}))\right) \\
\mathrm{O}_{e p} &= \delta_{\text {domain }} \otimes \mathrm{E}
\end{aligned}
$$

场景侧特征包括场景id、user和item在该场景下的统计量信息，比如曝光、点击等。

#### Parameter Personalized Network(PPNet)
对多任务DNN参数施加样本粒度的个性化影响，来选择和增强用户对于该任务的个性化参数信号，也就是在网络中注入用户在某个场景某类交互行为的倾向性先验：
$$
\begin{aligned}
\mathbf{O}_{\text {prior }} &= E\left(\mathcal{F}_u\right) \oplus E\left(\mathcal{F}_i\right) \oplus E\left(\mathcal{F}_a\right) \\
\delta_{\text {task }} &= \mho_{\text {pp }}\left(\mathbf{O}_{\text {prior }} \oplus\left(\oslash\left(\mathbf{O}_{e p}\right)\right)\right) \\
\mathbf{O}_{p p} &= \boldsymbol{\delta}_{\text {task }} \otimes \mathbf{H} \\
\mathbf{O}_{p p}^{(l)} &= \boldsymbol{\delta}_{\text {task }}^{(l)} \otimes \mathbf{H}^{(l)} \\
\mathbf{H}^{(l+1)} & =f\left(\mathbf{O}_{p p}^{(l)} \mathbf{W}^{(l)}+\boldsymbol{b}^{(l)}\right), l \in\{1, \ldots, L\}
\end{aligned}
$$

- $\mathcal{F}_u, \mathcal{F}_i, \mathcal{F}_a$分别表示user、item、author侧特征
- $\mathbf{O}_{e p}$需要做梯度冻结，这样PPNet网络的更新不会影响EPNet，避免跷跷板问题

## 实验结果


![exp](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.3rbd8x840o.webp)

在各场景、各任务中表现优异，均为sota，PEPNet建模的效果确实显著。

___

## 参考
