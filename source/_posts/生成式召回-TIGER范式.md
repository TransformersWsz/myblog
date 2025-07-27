---
title: 生成式召回-TIGER范式
mathjax: true
toc: true
date: 2025-07-27 22:29:48
updated: 2025-07-27 22:29:48
categories:
- 搜广推
tags:
- LLM
- Retrieval
- Transformer
- RQ-VAE
---

下面是 TIGER（Transformer Index for GEnerative Recommenders）这篇 Google 提出的 NeurIPS 2023 论文《Recommender Systems with Generative Retrieval》 的详细讲解，以 Markdown 博客格式进行整理，方便你直接复制使用 📘

<!--more-->

### 📌 背景与痛点

* 传统推荐系统往往是 **先双塔 embedding 近似检索**（ANN），再进行排序；
* 这种方式存在：

  * Embedding 表过大、内存和存储开销大；
  * 新品/冷启动 item 没有 embedding，难以推荐；
  * 难以通过模型结构共享相似 item 的语义信息，从而限制泛化性 ([Shashank Rajput][2])。

---

### 🧠 核心创新点（TIGER 模型亮点）

1. **生成式检索（Generative Retrieval）**
   通过自回归解码生成目标 item 的 ID，而不是传统 embedding + ANN。Transformer 的 decoder 直接输出 item 的“Semantic ID”作为推荐结果 ([NeurIPS Papers][3])。

2. **Semantic ID 表示**

   * 使用内容编码（如 SentenceT5）生成 item embedding；
   * 将 embedding 经 **Residual Quantization VAE（RQ‑VAE）** 量化为一系列 codeword Tuple，即 Semantic ID；
   * 各 token 有语义意义，编码符号总量远小于 item 总量，但语义可区分 item ([NeurIPS Papers][3])。

3. **端到端 Seq2Seq 架构**

   * **Bidirectional Transformer Encoder** 汇聚用户交互历史的 Semantic ID token 序列；
   * **Autoregressive Transformer Decoder** 生成下一个 item 的 Semantic ID token 序列；
   * 从用户行为历史直接预测下一个 item，无需 embedding\_pair + ANN 检索 ([NeurIPS Papers][3])。

---

### ✅ TIGER 解决的痛点和优势

| 痛点                    | TIGER 的解决方法                                 | 优势                        |
| --------------------- | ------------------------------------------- | ------------------------- |
| embedding 太大 / 存储高    | Semantic ID token 数量极小，token vocabulary 可控制 | 内存友好、减小表规模                |
| 冷启动 item embedding 缺失 | Semantic ID 来源于 item 内容特征                   | 可推广至新 item，无需训练 embedding |
| 类似 item 无共享           | 相似内容生成相近的 Semantic ID                       | 用户语义共享，加强泛化               |
| 模型检索复杂                | Transformer decoder 直接生成                    | 端到端简洁流程                   |

---

### 🧪 实验 & 结果

* 在多个 **公开 benchmark 数据集**（如 Amazon 等）上，TIGER **显著优于传统 SOTA 模型**，包括 recall 和 NDCG 指标 ([arXiv][4], [NeurIPS Papers][3], [arXiv][1], [Shashank Rajput][2])。
* 特别表现：

  * **冷启动 item 推荐** 显著提升；
  * 使用可控解码（如 beam search 长度、sampling diversity）可以方便地调节推荐多样性；
  * 与纯随机 ID 或 LSH ID（即非语义 token）相比，RQ‑VAE Semantic ID 带来更高的效果 ([NeurIPS Papers][3], [Shashank Rajput][2])。

---

### 📝 结构示意图（Markdown 支持的渲染）

```text
User session sequence:  t_5 → t_23 → t_78  (Semantic ID tokens of visited items)
        │
    Encoder (bidirectional Transformer)
        ↓
   Context embedding
        →
   Decoder (autoregressive Transformer) outputs: ⟨BOS⟩ t_25 t_55 ⟨EOS⟩
        → Lookup item with (t_25, t_55) → next recommended item
```

* Semantic ID 元素如：Item 515 → (5, 25, 78)，Item 233 → (5, 23, 55)；
* Embedding 经 RQ‑VAE 量化成多个 level 的 codeword 构成 tuple；
* Decoder 解码这些 code 作为推荐目标。

---

### 🧾 总结

* TIGER 是第一篇将 **Generative Retrieval 自回归生成方式** 应用于推荐系统的工作；
* 它通过 **Semantic ID 和 Seq2Seq Transformer**，突破 embedding + ANN 的传统限制；
* 在 **冷启动、多样性、效率和泛化能力**上展现强优势；
* 适用于大规模推荐场景，尤其是 content-rich、item 海量、频繁上线新品的平台。

---

### 🔗 如果你写博客，可以这样引用本论文

> Rajput et al. (2023) 在 NeurIPS 2023 提出了 TIGER 模型，通过将推荐任务转化为一个生成 next-item Semantic ID 的序列预测问题，开启了推荐系统领域的新范式——将 Transformer 的生成能力当作推荐索引，替代传统 embedding + ANN 检索机制 ([NeurIPS Papers][3], [OpenReview][5])。

---

如需我提供实验表格、参数设置、pipeline 细节等补充内容，也可以继续告诉我\~

[1]: https://arxiv.org/abs/2305.05065?utm_source=chatgpt.com "[2305.05065] Recommender Systems with Generative Retrieval - arXiv"
[2]: https://shashankrajput.github.io/Generative.pdf?utm_source=chatgpt.com "[PDF] Recommender Systems with Generative Retrieval - Shashank Rajput"
[3]: https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf?utm_source=chatgpt.com "[PDF] Recommender Systems with Generative Retrieval"
[4]: https://arxiv.org/abs/2411.18814?utm_source=chatgpt.com "Unifying Generative and Dense Retrieval for Sequential Recommendation"
[5]: https://openreview.net/forum?id=BJ0fQUU32w&utm_source=chatgpt.com "Recommender Systems with Generative Retrieval - OpenReview"
