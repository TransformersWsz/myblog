---
title: 常见的LLM推理加速解决方案
mathjax: true
toc: true
date: 2023-12-03 17:32:15
categories:
- NLP
tags:
- LLM
- Transformer
- 推理加速
---

- KV Cache
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