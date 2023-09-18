---
title: Multi Query Attention & Group Query Attention
mathjax: true
toc: true
date: 2023-09-13 00:45:05
categories:
- NLP
tags:
- LLM
- Transformer
- 推理加速
---
Multi Query Attention(MQA)在2019年就被提出来了，用于推理加速，但在当时并没有受到很多关注，毕竟一张2080就能跑Bert-base了。随着LLM的大火，MQA所带来的收益得以放大。

<!--more-->

## 思路

Multi Query Attention(MQA)跟Multi Head Attention(MHA)只有一词之差，但其思路非常简单，几乎跟MHA一致：

![model](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.670koriphig0.webp)


MHA的Query、Key、Value分拆成8个头，每个头进行self-attention运算，而MQA是Query分成8个头，每个头共享一组Key和Value

```
MHA: Q, K, V = (512, 768), # seq_len, hidden_dim
			拆成8个头：
			Q : (8, 512, 96) 
			k, v: (8, 512, 96)
MQA: 
 Q -> (512, 768) 
 K -> (512, 96)
 v -> (512, 96)
把Q拆成8个头：
Q： (8, 512, 96)
K, V：(512, 96)
```

## 代码实现
- MHA
```python
...
self.Wqkv = nn.Linear( 
            d_model,
            d_model * 3,
            device=device,
        )
...
```
将 `d_model * 3` 拆成3个768维

- MQA
```python
...
self.Wqkv = nn.Linear( 
            d_model,
            d_model + 2 * self.head_dim,
            device=device,
        )
...
```
将 `d_model + 2 * self.head_dim` 拆成1个768维 + 2个96维

可以看到参数数量大幅减少。

## 实验结果
实验指标略微降低，但推理加速非常明显。

![result](https://raw.githubusercontent.com/TransformersWsz/image_hosting/master/image.194dl27xykcg.webp)


## Group Query Attention
Q拆分成8个头，K和V分别拆成4个头，然后对应进行attention运算。
___


## 参考
- [Fast Transformer Decoding: One Write-Head is All
You Need](https://arxiv.org/pdf/1911.02150.pdf)
- [[LLM] multi query attention加速推理解码](https://zhuanlan.zhihu.com/p/645808819)