---
title: excel根据百分比绘制数据条
mathjax: true
toc: true
date: 2026-11-20 01:06:45
updated: 2026-11-20 01:06:45
categories:
- 随笔
tags:
- Excel
- 数据条
---
我们可以用数据条让百分比数据看起来更加直观。下面记录操作步骤：

<!--more-->

1. 数据准备：工号为第一列，第二列展示每个员工的完工率
![1](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.48230c.png)
1. 将小数调整成百分比形式
![2](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.40369d.png)
1. 调用 `条件格式 > 数据条`
![3.1](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.73a4a9.png)
![3.2](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.864b69.png)
1. 校准基线：由于数据条默认以最大值为满格（上图的245%），需要调整`1.0`为基线

*步骤1：管理规则*
![4.1](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.aa6591.png)
*步骤2：编辑规则*
![4.2](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.c0603c.png)
*步骤3：编辑最大值*
![4.3](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.c22435.png)
1. 效果展示
![5](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.245fb3.png)

