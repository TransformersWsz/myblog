---
title: MR编程注意事项
mathjax: true
toc: true
date: 2022-09-06 23:21:11
categories:
- Software Engineering
tags:
- MapReduce
- Big Data
---
在公司集群上跑MapReduce的时候会遇到一些异常报错，主要还是我们编程时没注意极端情况，想当然的认为没有bug就能顺利运行。以下列举几种例子：

<!--more-->

## Reduce卡在某个进度
```java
while (iterator.hasNext()) {
	System.out.prinltn("Hello World");
    // String[] arr = iterator.next().toString().split("\t");
}
```
这是因为没有进行 `iterator.next()` 操作，导致程序陷入死循环。如果其中还有写数据的逻辑，那么可能导致磁盘空间紧张。

## Inner error, IOException
如果单独拉一个part下来能测试通过，但在集群上老是报上述错误，那么有两种情况：
- 相同key下的value内的元素过多，有千万个
- 不同的key太多，有千万个

上述两种情况不一定会触发异常报错，但如果出现了，请从这两个方面排查。