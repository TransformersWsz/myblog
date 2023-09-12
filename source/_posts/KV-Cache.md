---
title: KV Cache
mathjax: true
toc: true
date: 2023-09-13 02:05:38
categories:
- NLP
tags:
- LLM
- Transformer
- 推理加速
- KV Cache
---

大模型推理加速的一个常用技术是KV Cache，在不牺牲任何计算精度的前提下，通过空间换时间，提高推理性能。注意，这里的Cache概念非常简单，跟浏览器缓存、CPU缓存不是一个概念。

<!--more-->

在生成式模型的推理过程中，假设给定一个输入文本，模型会输出一个长度为N的文本，但是该过程执行了N次推理。因为模型每次推理只输出一个token，然后将输出token与输入tokens拼接在一起，作为下一次推理的输入，这样不断反复直到遇到终止符。

由于生成式模型推理过程是单向的，即已经输出的token的embedding是不会再变化的，所以上述步骤可以优化。将Key和Value缓存起来，不用再经历前向传播算出embedding，只需要将上一轮输出的token前向传播算出embedding，然后与KV拼接，来预测出下一个token。这样模型的计算量大大减少，推理大幅加速。

伪代码如下：
```python
query = self._split_heads(query, self.num_heads, self.head_dim)
key = self._split_heads(key, self.num_heads, self.head_dim)
value = self._split_heads(value, self.num_heads, self.head_dim)

if layer_past is not None: # 当输出第一个token后，layer_past就是非None了
    past_key, past_value = layer_past # 取出之前计算好的 key, value
    key = torch.cat((past_key, key), dim=-2) # past_key 与当前 token 对应的 key 拼接
    value = torch.cat((past_value, value), dim=-2) # past_value 与当前 token 对应的 value 拼接

if use_cache is True:
    present = (key, value)
else:
    present = None
```

___

## 参考
- [KV Cache](https://www.zhihu.com/question/596900067/answer/3040011798)