---
title: Dataset
mathjax: true
date: 2019-07-28 00:12:01
categories:
- NLP
tags:
- 论文阅读
- Data
---

Here are some detailed descriptions of datasets for NLP experiments:

<!--more-->

## MNLI

Multi-Genre Natural Language Inference is a large-scale, crowdsourced entailment classification task. Given a pair of sentences, the goal is to predict whether the second sentence is an entailment, contradiction, or neutral with respect to the first one.

## QQP

Quora Question Pairs is a binary classification task where the goal is to determine if two questions asked on Quora are semantically equivalent.

## QNLI

Question Natural Language Inference is a version of the Stanford Question Answering Dataset which has been converted to a binary classification task. The positive examples are (question, sentence) pairs which do contain the correct answer, and the negative examples are (question, sentence) from the same paragraph which do not contain the answer.

## SST-2

The Stanford Sentiment Treebank is a binary single-sentence classification task consisting of sentences extracted from movie reviews with human annotations of their sentiment.

## CoLA

The Corpus of Linguistic Acceptability is a binary single-sentence classification task, where the goal is to predict whether an English sentence is linguistically “acceptable” or not.

## STS-B

The Semantic Textual Similarity Benchmark is a collection of sentence pairs drawn from news headlines and other sources. They were annotated with a score from 1
to 5 denoting how similar the two sentences are in terms of semantic meaning.

## MRPC

Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.

## RTE

Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with
much less training data.

## SQuAD v1.1

The Stanford Question Answering Dataset (SQuAD v1.1) is a collection of 100k crowdsourced question/answer pairs. Given a question and a passage from Wikipedia containing the answer, the task is to
predict the answer text span in the passage.

## SQuAD v2.0

The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic.

## SWAG

The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair completion examples that evaluate grounded commonsense inference. Given a sentence, the task is to choose the most plausible continuation among four choices.
