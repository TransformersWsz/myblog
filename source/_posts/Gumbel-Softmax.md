---
title: Gumbel Softmax
mathjax: true
toc: true
date: 2024-04-07 23:08:23
updated: 2024-04-07 23:08:23
categories:
- Machine Learning
tags:
- Argmax
- Softmax
- 可导
---
Argmax是不可求导的，Gumbel Softmax允许模型能从网络层的离散分布（比如类别分布categorical distribution）中稀疏采样的这个过程变得可微，从而允许反向传播时可以用梯度更新模型参数。

<!--more-->

## 算法流程
1. 对于某个网络层输出的 $\mathrm{n}$ 维向量 $v=\left[v_1, v_2, \ldots, v_n\right]$，生成 $\mathrm{n}$ 个服从均匀分布 $\mathrm{U}(0,1)$ 的独立样本 $\epsilon_1, \ldots, \epsilon_n$
2. 通过 $G_i=-\log \left(-\log \left(\epsilon_i\right)\right)$ 计算得到 $G_i$
3. 对应相加得到新的值向量 $v^{\prime}=\left[v_1+G_1, v_2+G_2, \ldots, v_n+G_n\right]$
4. 通过softmax函数计算各个类别的概率大小，其中 $\tau$ 是温度参数：
$$
p_\tau\left(v_i^{\prime}\right)=\frac{e^{v_i^{\prime} / r}}{\sum_{j=1}^n e^{v_j^{\prime} / \tau}}
$$

## Gumbel-Max Trick
Gumbel分布是专门用来建模从其他分布（比如高斯分布）采样出来的极值形成的分布，而我们这里“使用argmax挑出概率最大的那个类别索引”就属于取极值的操作，所以它属于Gumbel分布。

{% note danger %}
注意，极值的分布也是有规律的。
{% endnote %}

Gumbel-Max Trick的采样思想：先用均匀分布采样出一个随机值，然后把这个值带入到gumbel分布的CDF函数的逆函数得到采样值，即我们最终想要的类别索引。公示如下：
$$
z=\operatorname{argmax}_i\left(\log \left(p_i\right)+g_i\right) \\
g_i=-\log \left(-\log \left(u_i\right)\right), u_i \sim U(0,1)
$$
上式使用了重参数技巧把采样过程分成了确定性的部分和随机性的部分，我们会计算所有类别的log分布概率（确定性的部分），然后加上一些噪音（随机性的部分），这里噪音是标准gumbel分布。在我们把采样过程的确定性部分和随机性部分结合起来之后，我们在此基础上再用一个argmax来找到具有最大概率的类别。

## Softmax
使用softmax替换不可导的argmax，用温度系数 $\tau$ 来近似argmax：
$$
p_i^{\prime}=\frac{\exp \left(\frac{g_i+\log p_i}{\tau}\right)}{\sum_j \exp \left(\frac{g_j+\log p_j}{\tau}\right)}
$$
$\tau$ 越大，越接近argmax。

___

## 参考
- [CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX](https://openreview.net/pdf?id=rkE3y85ee)
- [通俗易懂地理解Gumbel Softmax](https://zhuanlan.zhihu.com/p/633431594)