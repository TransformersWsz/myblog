---
title: expand和repeat区别
mathjax: true
date: 2020-12-08 22:06:34
categories:
- pytorch学习笔记
tags:
- tensor
---

`expand` 和 `repeat` 都是对张量进行扩维，这里记录下使用区别。

<!--more-->

# expand()

> Returns a new view of the :attr:`self` tensor with singleton dimensions expanded to a larger size.

将张量`=1`的维度进行扩展，`>1`的维度保持不变。


```python
import torch
```


```python
a = torch.tensor([[12, 3, 4]])
a.expand(3,-1)
```




    tensor([[12,  3,  4],
            [12,  3,  4],
            [12,  3,  4]])



# repeat()
> Repeats this tensor along the specified dimensions.

将张量沿着特定维度进行复制。


```python
b = torch.tensor([[1,2,4],[4,5,6]])
b.repeat(3,2)
```




    tensor([[1, 2, 4, 1, 2, 4],
            [4, 5, 6, 4, 5, 6],
            [1, 2, 4, 1, 2, 4],
            [4, 5, 6, 4, 5, 6],
            [1, 2, 4, 1, 2, 4],
            [4, 5, 6, 4, 5, 6]])
