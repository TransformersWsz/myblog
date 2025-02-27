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

![Adapter](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.5xarunoecl.webp)

用户在多场景、多任务下的行为存在共性和差异性，如何用联合建模来捕捉这些特性又避免跷跷板效应成为一大难点。


## 模型结构