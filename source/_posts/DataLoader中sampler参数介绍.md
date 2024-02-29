---
title: DataLoader中sampler参数介绍
mathjax: true
toc: true
date: 2021-12-24 14:42:06
updated: 2021-12-24 14:42:06
categories:
- PyTorch
tags:
- DataLoader
- sampler
---

`Sampler` 决定了 `Dataset` 的采样顺序。

<!--more-->

 ## `DataLoader` | `Sampler` | `DataSet` 关系

{% asset_img relation.png %}


- `Sampler` : 提供数据集中元素的索引
- `DataSet` : 根据 `Sampler` 提供的索引来检索数据
- `DataLoader` : 批量加载数据用于后续的训练和测试

## `Sampler`

```python
class Sampler(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
```

PyTorch官网已经实现了多种 `Sampler` :

### `SequentialSampler`

> 若 `shuffle=False` ，且未指定 `sampler` ，默认使用

```python
class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
```

### `RandomSampler`

> 若 `shuffle=True` ，且未指定 `sampler` ，默认使用

```python
class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source):
        self.data_source = data_source


    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples
```

### `BatchSampler`

> like `sampler`, but returns a batch of indices at a time. Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`

- 在 `DataLoader` 中设置 `batch_sampler=batch_sampler` 的时候，上面四个参数都必须是默认值。也很好理解，每次采样返回一个batch，那么 `batch_size` 肯定为 `1`

```python
class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```

- 可以看到在构造 `BatchSampler` 实例的时候，需要传入一个sampler作为实参
___

## 最佳实践

最近看到一篇推文，分享了一个使模型训练速度提升20%的Trick--[BlockShuffle](https://mp.weixin.qq.com/s/xGvaW87UQFjetc5xFmKxWg) 。fork了原作者的代码，并自定义了 `batch_sampler` ，源码见：[TransformersWsz/BlockShuffleTest](https://github.com/TransformersWsz/BlockShuffleTest)

___

## 参考

- [一个使模型训练速度提升20%的Trick--BlockShuffle](https://mp.weixin.qq.com/s/xGvaW87UQFjetc5xFmKxWg)
- [Pytorch DataLoader详解](https://www.zdaiot.com/MLFrameworks/Pytorch/Pytorch%20DataLoader%E8%AF%A6%E8%A7%A3/)
- [torch.utils.data — PyTorch 1.10.1 documentation](https://PyTorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)
- [PyTorch中用Mnist数据集dataloader 自定义batchsampler - 代码先锋网 (codeleading.com)](https://www.codeleading.com/article/79575865698/)
- [PyTorch 实现一个自定义的dataloader，每个batch都可以实现类别数量均衡 (tqwba.com)](https://www.tqwba.com/x_d/jishu/415752.html)
- [一文弄懂Pytorch的DataLoader, DataSet, Sampler之间的关系](https://zhuanlan.zhihu.com/p/76893455)
