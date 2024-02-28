---
title: python2字典遍历方法性能对比
mathjax: true
toc: true
date: 2022-07-21 22:54:28
categories:
- 编程语言
tags:
- Python2
- Dict
- tqdm
---
在公司服务器上跑python2程序时，使用了 `tqdm(d.items())` 来包裹字典，发现进度条一直卡在0%不动，怀疑是 `d.items()` 取出所有的元素作为列表返回，而不是迭代器，导致耗时非常长。在此做一下性能测试，代码如下：

<!--more-->

```python
import random
import time
from tqdm import tqdm


def construct_large_dict(length=25000000, size=3):
    d = {}
    for i in tqdm(range(length)):
        l = []
        for _ in range(size):
            l.append(random.randint(0, 100))
        d[i] = l
    return d


if __name__ == "__main__":
    d = construct_large_dict()

    total = 0
    start = time.time()
    for key in d:
        total += sum(d[key])
    end1 = time.time()
    print("method-1 time: {}".format(end1 - start))

    total = 0
    for k, v in d.items():
        total += sum(v)
    end2 = time.time()
    print("method-2 time: {}".format(end2 - end1))

    total = 0
    for k, v in d.iteritems():
        total += sum(v)
    end3 = time.time()
    print("method-3 time: {}".format(end3 - end2))
  
```
结果如下：
![result](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.1zhzwi7qjc.webp)

因此在遍历大字典时，推荐使用第一种或者第三种方式，进度条展示的时候也更人性化。

> 在python3中，已经用 `d.items()` 代替了 `d.iteritems()` ，因此无需再担心性能问题。