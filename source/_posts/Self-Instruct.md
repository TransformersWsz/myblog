---
title: Self-Instruct
mathjax: true
toc: true
date: 2023-10-11 02:19:51
categories:
- Machine Learning
tags:
- LLM
- Instruction
---

本篇工作利用LLM的生成能力，来产生大量指令数据集（指令、输入、输出），无需人工标注数据。

<!--more-->

![flow](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.4zfugmx2oqc0.webp)

其中，在对任务判别的时候，需要区分是输出优先还是输入优先。
- 输入优先没问题，符合人类直觉，给定指令和输入，然后产生输出。
- 当任务是分类任务的时候，采用输出优先，即先生成一个标签，然后根据标签生成相应的输入文本。这是因为分类任务，如果输入优先，模型倾向于生成正确的文本，比如语法正确的语句，不会产生错误的语句。因此先给出标签“错误”，强制模型根据错误标签生成错误的语句

根据LLM生成的指令来微调LLM，更多是为了提升LLM在零样本任务上的泛化能力：

![ret](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.6x9fo3xko5s0.webp)

千万不要误解成了模型自己生成输入和标签，然后自己学习，自娱自乐。