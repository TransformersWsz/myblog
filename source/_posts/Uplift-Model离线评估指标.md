---
title: Uplift Model离线评估指标
mathjax: true
toc: true
date: 2024-09-16 16:28:35
updated: 2024-09-16 16:28:35
categories:
- 营销
tags:
- AUUC
- Uplift Curve
- Qini coefficient
- Qini Curve
---
uplift建模难点在于无法获得个体的ground truth，因为它是反事实的。只能通过构造treatment和control两组镜像人群，对比两组人群的转化增量，来实现模型性能的评估。

<!--more-->

## Uplift Curve

#### 计算公式
$$
f(k)=\left(\frac{Y_k^T}{N_k^T}-\frac{Y_k^C}{N_k^C}\right)\left(N_k^T+N_k^C\right)
$$

具体计算步骤如下：
1. 模型对样本集预测，然后将样本按照预测得到的uplift value进行降序排序
2. 取topK个样本，计算得到 $f(k)$ 。以 $k$ 为横轴，$f(k)$ 为纵轴，画出Uplift Curve
   - $Y_k^T$ 表示topK个样本中， treatment组有转化的样本数，$Y_k^C$同理
   - $Y_k^T$ 表示topK个样本中， treatment组有转化的样本数，$Y_k^C$同理
3. Uplift Curve下的面积即是AUUC