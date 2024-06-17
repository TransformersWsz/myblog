---
title: LLaMA2详解
mathjax: true
toc: true
date: 2024-06-02 02:19:35
updated: 2024-06-02 02:19:35
categories:
- NLP
tags:
- LLM
- LLaMA
---

LLaMA2的模型结构拆解：

<!--more-->

![llama](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.2krpqrv609.png)

## RoPE
这个不多谈了，见：[旋转位置编码](https://transformerswsz.github.io/2023/09/04/%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81/)

## RMSNorm
标准Transformer的LayerNorm如下：
$$
y=\frac{x-\operatorname{Mean}(x)}{\sqrt{\operatorname{Var}(x)+\varepsilon}} * W+B
$$

llama2采用了RMSNorm：
$$
y=\frac{x}{R M S(x)+\varepsilon}, RMS(x) = \sqrt{\frac{1}{n} \sum_{1}^n {x_i}^2}
$$

[实现源码](https://github.com/meta-llama/llama/blob/b8348da38fde8644ef00a56596efb376f86838d1/llama/model.py#L52)：
```python
def _norm(self, x):
    """
    Apply the RMSNorm normalization to the input tensor.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The normalized tensor.

    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

RMSNorm能保证激活效果变化，且计算效率能提升7%∼64%

## SwiGLU
这个不多谈了，见：[SwiGLU激活函数](https://transformerswsz.github.io/2024/05/09/SwiGLU%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/)

## MLP
![MLP](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.7lju1gc3hj.webp)

[up、gate、down都是三个linear层](https://github.com/meta-llama/llama/blob/b8348da38fde8644ef00a56596efb376f86838d1/llama/model.py#L307)：`down(up * silu(gate))`

```python
def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

## GQA
这个不多谈了，见：[Multi Query Attention & Group Query Attention](https://transformerswsz.github.io/2023/09/13/Multi-Query-Attention-Group-Query-Attention/)

## padding_side
llama系列训练和推理都是right padding：

- 训练：其实只要设置padding mask，那么left/right pad是没有区别的。然而实际上huggingface中某些tokenizer在training的时候必须设成right padding，因为有些tokenizer使用的是absolute position id，导致非padding token的position id不是从0开始的。
- 推理：由于llama是decoder-only结构，每个token的生成依赖于上一个token。而上一个token如果是无实际意义的padding token，那么就打断了句子语义的连续性，导致生成文本质量较差。因此left padding比较make sense，但在llama推理的源码中，batch的时候采用[right padding](https://github.com/meta-llama/llama3/blob/14aab0428d3ec3a9596f1dea06d9c564f9c0e35f/llama/generation.py#L155)。生成token的时候，[从batch中长度最短的序列开始生成，其它较长序列的相同位置保持不变](https://github.com/meta-llama/llama3/blob/14aab0428d3ec3a9596f1dea06d9c564f9c0e35f/llama/generation.py#L184)，直到该序列开始生成。猜测llama这样做的原因是保持跟训练一致，但不如left padding优雅。

## llama1,llama2,llama3区别
|差异|llama1|llama2|llama3|
|:---:|:---:|:---:|:---:|
|上下文长度|2k|4k|8k|
___

## 参考
- [大部分的大模型(LLM)采用左填充(left-padding)的原因](https://zhuanlan.zhihu.com/p/646852375)
- [从头预训练一只超迷你 LLaMA 3](https://mp.weixin.qq.com/s/Yf_NU3pgedLHl8dWAaMRfQ)
- [LLM padding 细节](https://zhuanlan.zhihu.com/p/675273498)
- [LLM - Transformer && LLaMA2 结构分析与 LoRA 详解](https://blog.csdn.net/BIT_666/article/details/132161203)