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
Argmax是不可求导的，Gumbel Softmax允许模型能从神经元的离散分布（比如类别分布categorical distribution）中采样的这个过程变得可微，从而允许反向传播时可以用梯度更新模型参数。

<!--more-->

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
上式使用了重参数技巧把采样过程分成了确定性的部分和随机性的部分，我们会计算所有类别的log分布概率（确定性的部分），然后加上一些噪音（随机性的部分），上面的例子中，噪音是标准高斯分布，而这里噪音是标准gumbel分布。在我们把采样过程的确定性部分和随机性部分结合起来之后，我们在此基础上再用一个argmax来找到具有最大概率的类别
___

## 参考
- [通俗易懂地理解Gumbel Softmax](https://zhuanlan.zhihu.com/p/633431594)