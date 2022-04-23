---
title: pytorch中的二分类及多分类交叉熵损失函数
mathjax: true
date: 2020-12-08 21:55:51
categories:
- pytorch学习笔记
tags:
- 损失函数
- pytorch
---

本文主要记录一下pytorch里面的二分类及多分类交叉熵损失函数的使用。

<!--more-->
___


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(2020)
```




    <torch._C.Generator at 0x7f4e8b3298b0>



## 二分类交叉熵损失函数
#### Single


```python
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
print(input)
target = torch.empty(3).random_(2)
output = loss(m(input), target)
print(output)
f_output = F.binary_cross_entropy(m(input), target)
print(f_output)
l_output = nn.BCEWithLogitsLoss()(input, target)
print(l_output)
```

    tensor([ 1.2372, -0.9604,  1.5415], requires_grad=True)
    tensor(0.2576, grad_fn=<BinaryCrossEntropyBackward>)
    tensor(0.2576, grad_fn=<BinaryCrossEntropyBackward>)
    tensor(0.2576, grad_fn=<BinaryCrossEntropyWithLogitsBackward>)


#### Batch


```python
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(32,5, requires_grad=True)
target = torch.empty(32,5).random_(2)
output = loss(m(input), target)
print(output)
f_output = F.binary_cross_entropy(m(input), target)
print(f_output)
l_output = nn.BCEWithLogitsLoss()(input, target)
print(l_output)
```

    tensor([[ 1.2986,  1.5832, -1.1648,  0.8027, -0.9628],
            [-1.5793, -0.2155,  0.4706, -1.2511,  0.7105],
            [-0.1274, -1.9361,  0.8374,  0.0081, -0.1504],
            [ 0.1521,  1.1443,  0.2171, -1.1438,  0.9341],
            [-3.3199,  1.2998,  0.3918,  0.8327,  1.2411],
            [-0.8507, -0.1016, -1.2434, -0.5755,  0.1871],
            [-0.3064,  1.3751,  1.8478,  0.0326,  0.2032],
            [ 0.1782,  2.3037,  1.5948, -1.4731,  1.5312],
            [-0.9075, -1.7135,  0.4650, -1.7061,  0.0625],
            [-1.1904,  0.1130, -1.6609, -0.2000, -0.1422],
            [ 0.3307, -0.8395, -1.3068, -0.8891,  0.9858],
            [ 0.5484,  0.7461, -1.0738, -2.2162,  0.6801],
            [-0.8803,  0.9934, -1.6438,  0.3860,  0.4111],
            [-1.1078, -0.9629, -0.9534, -0.6207,  0.6885],
            [-0.0175,  1.9496,  0.9740, -0.4687, -0.6127],
            [ 0.3713,  0.8074,  0.3072,  1.1604, -0.2669],
            [-0.1773, -0.2787,  0.1926,  0.7492,  0.7492],
            [-0.3126, -0.3321, -1.7287, -3.0126,  0.1194],
            [ 1.0486, -0.1890, -0.5853,  0.4353,  0.2619],
            [ 1.9726, -0.5510, -0.1826, -0.8600, -0.9906],
            [ 0.7551,  0.8431, -0.8461, -1.2120,  0.2908],
            [-0.0932, -0.7151, -0.0631,  1.7554,  0.7374],
            [-0.1494, -0.6990, -0.1666,  2.0430,  1.3968],
            [ 0.2280, -0.3187,  1.0309, -0.1067,  1.1622],
            [-1.5120, -0.8617,  1.4165, -0.2361, -0.0355],
            [-0.8757, -0.6554,  0.1121, -0.1669, -0.2628],
            [-0.8023,  0.2305, -1.1792,  0.4314, -0.3653],
            [ 0.7487,  0.5358, -0.2677, -0.8128,  0.3029],
            [ 1.4439, -0.5677,  0.5564, -0.2485, -0.3281],
            [-2.0259,  1.1038,  1.0615,  1.7317, -0.0531],
            [ 0.9083, -0.8274,  0.8101, -1.1375, -1.2009],
            [ 0.3300, -0.8760,  1.3459, -1.0209, -0.5313]], requires_grad=True)
    tensor(0.8165, grad_fn=<BinaryCrossEntropyBackward>)
    tensor(0.8165, grad_fn=<BinaryCrossEntropyBackward>)
    tensor(0.8165, grad_fn=<BinaryCrossEntropyWithLogitsBackward>)


### Note

- `nn.BCELoss()` 与 `F.binary_cross_entropy` 计算结果是等价的，具体两者差距可见[PyTorch 中，nn 与 nn.functional 有什么区别？](https://www.zhihu.com/question/66782101)
- > `nn.BCEWithLogitsLoss`: combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability. 至于为什么更稳定，见 https://blog.csdn.net/u010630669/article/details/105599067
- 二分类交叉熵损失函数的input和target的shape是一致的

## 多分类交叉熵损失函数
#### Single


```python
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(output)
f_output = F.cross_entropy(input, target)
print(f_output)
```

    tensor(1.7541, grad_fn=<NllLossBackward>)
    tensor(1.7541, grad_fn=<NllLossBackward>)


#### Batch


```python
loss = nn.CrossEntropyLoss()
input = torch.randn(32, 10, 5, requires_grad=True)
target = torch.empty(32, 5, dtype=torch.long).random_(10)
output = loss(input, target)
print(output)
f_output = F.cross_entropy(input, target)
print(f_output)
```

    tensor(2.7944, grad_fn=<NllLoss2DBackward>)
    tensor(2.7944, grad_fn=<NllLoss2DBackward>)


### Note

- `nn.CrossEntropyLoss` 与 `F.cross_entropy` 计算结果是等价的。两个函数都结合了 `LogSoftmax` and `NLLLoss` 运算
- [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss) 的公式为：$\operatorname{loss}(\mathrm{x}, \text { class })=-\log \left(\frac{\exp (\mathrm{x}[\mathrm{class}])}{\sum_{\mathrm{j}} \exp (\mathrm{x}[\mathrm{j}])}\right)=-\mathrm{x}[\mathrm{class}]+\log \left(\sum_{\mathrm{j}} \exp (\mathrm{x}[\mathrm{j}])\right)$,这与我们平时见到的多分类交叉熵损失函数有点不同，具体的推导过程见[Pytorch里的CrossEntropyLoss详解](https://www.cnblogs.com/marsggbo/p/10401215.html)

___
## 参考

- [Docs > torch.nn > CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss)
- [Docs > torch.nn > BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html?highlight=bceloss#torch.nn.BCELoss)
- [Pytorch里的CrossEntropyLoss详解](https://www.cnblogs.com/marsggbo/p/10401215.html)

