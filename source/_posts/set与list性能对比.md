---
title: set与list性能对比
mathjax: true
toc: true
date: 2022-07-24 15:29:57
categories:
- 编程语言
tags:
- Python
- set
- list
---

当集合中的数据量特别大时，要判断一个元素是否在该集合中，建议使用 `set` 而不是 `list` ，两种性能差异非常大。下面做一个测试：

<!--more-->

```python
from tqdm import tqdm
import random
import time
```

## 构造长度为 `length` 的数组


```python
def construct_data(length):
    l = []
    for i in range(length):
        l.append(i)
    random.shuffle(l)
    return l, set(l)
```

## 测试 `num` 次

#### 测试 `list`


```python
length, num = int(1e6), int(1e4)
l, s = construct_data(length)
start_l = time.time()
for _ in tqdm(range(num)):
    r = random.randint(0, length-1)
    if r in l:
        pass
end_l = time.time()
print("test list time: {} seconds".format(end_l-start_l))
```

    100%|██████████| 10000/10000 [02:52<00:00, 58.00it/s]
    
    test list time: 172.42421102523804 seconds


​    


#### 测试 `set`


```python
start_s = time.time()
for _ in tqdm(range(num)):
    r = random.randint(0, length-1)
    if r in s:
        pass
end_s = time.time()
print("test set time: {} seconds".format(end_s-start_s))
```

    100%|██████████| 10000/10000 [00:00<00:00, 343595.45it/s]
    
    test set time: 0.03251051902770996 seconds


​    

可以看到，`set` 的速度实在比 `list` 快很多。毕竟 `set` 底层用hash散列实现，查找一个元素理论上只需 `O(1)` 时间，而 `list` 则是遍历，需要 `O(n)` 时间。数据量小的时候，两者看不出差距，数据量稍微大点，差距非常明显。
