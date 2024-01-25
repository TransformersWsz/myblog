---
title: 常见的LLM推理加速解决方案
mathjax: true
toc: true
date: 2024-01-26 17:32:15
categories:
- NLP
tags:
- LLM
- Transformer
- 推理加速
---

- KV Cache：用空间换时间
  - 当decoder输入序列是 $t_1, t_2, \dots, t_n$ 时，预测$t_{n+1}$，只需利用到 $q^n$ 以及历史所有的 $k^i, v^i, i \in \{1,\dots,n \}$ ：
  $$
    h_n = \sum_{i=1}^{n} softmax(q^n \cdot k^i) \cdot v^i \\
    t_{n+1} = f(h_n)
  $$
  无须冗余attention计算 $h_1, \dots, h_{n-1}$ 以及 qkv映射 $q_1=W_q(t_1), k_1=W_k(t_1), k_1=W_v(t_1), \dots, q_{n-1}=W_q(t_{n-1}), k_1=W_k(t_{n-1}), k_1=W_v(t_{n-1})$

<!--more-->

- int量化
- PagedAttention
- GQA
- Speculative Decoding
  - [code](https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L76)
  - [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/?utm_content=273712248&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024)
  - [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192.pdf)

___

## 参考
- [PyTorch造大模型“加速包”，不到1000行代码提速10倍！英伟达科学家：minGPT以来最好的教程式repo之一](https://mp.weixin.qq.com/s/sQJK8hO5L_SNczUaUXucJQ)
- [图解大模型推理优化之 KV Cache](https://mp.weixin.qq.com/s/6q2LmwoFG2LcN0iHoZjjqw)