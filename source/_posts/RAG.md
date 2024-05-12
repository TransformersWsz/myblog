---
title: RAG
mathjax: true
toc: true
date: 2024-05-12 14:20:49
updated: 2024-05-12 14:20:49
categories:
- NLP
tags:
- LLM
- RAG
---
现有的LLM已经具备了理解、生成、逻辑和记忆能力，RAG(Retrieval Augmented Generation)则是为其套上外挂，使LLM能够访问训练数据来源之外的权威知识库，并生成领域特定的内容，而无须重新训练模型。

<!--more-->

## RAG的优势
- 经济高效：LLM无须重新训练，即可访问和生成领域内容。
- 减轻幻觉：LLM根据用户输入，并根据它的训练语料生成内容。RAG引入了信息检索组件，该组件利用用户输入首先从新数据源提取信息。用户查询和相关信息都提供给LLM。LLM使用新知识及其训练数据来创建更好的响应。

## RAG的缺点
- 维护成本高：RAG需要实时维护其数据库，是个系统工程
- 平响增加：RAG增加了检索流程，使得响应耗时增加，影响用户体验

## RAG的工作流程

![RAG](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.4913czxecz.png)

___

## 参考
- [什么是 RAG？](https://aws.amazon.com/cn/what-is/retrieval-augmented-generation/)