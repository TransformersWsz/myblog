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
\mathbf{x}^{\prime} = Relu(xW+b) \\
\boldsymbol{\delta}=\gamma * \operatorname{Sigmoid}\left(\mathbf{x}^{\prime} \mathbf{W}^{\prime}+\boldsymbol{b}^{\prime}\right), \boldsymbol{\delta} \in[0, \gamma]
$$
