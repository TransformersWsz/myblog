---
title: STL学习
mathjax: true
toc: true
date: 2021-08-12 00:47:50
categories:
- C++
tags:
- 面试
- STL
---
记录一下常见的STL的概念和用法。

<!--more-->

## `vector`

`vector` 是一个自动扩容的容器，支持随机访问，底层通过动态数组实现。

#### 扩容
当 `vector` 执行 `insert` 或者 `push_back` 时，如果当容器的存储数量已经达到容量，就会触发扩容。具体流程如下：
1. 申请新的内存空间（原内存空间的1.5~2倍）
2. 把原空间的元素拷贝到新的空间里
3. 释放原空间
4. 数组指针指向新空间

#### 成员函数
- `size()` : 容器中元素的个数
- `capacity()` : 器在分配那块内存上可以容纳的元素的个数
- `resize(n)` : 强制将容器改为容纳为 `n` 个数据，分三种情况讨论：
    1. `n < size()` : 容器尾部元素被销毁
    2. `n > size()` : 新构造的元素会添加到末尾
    3. `n > capacity()` : 在元素加入前会进行重新分配
- `reserve(n)` : 强制容器把它的容量改为不小于 `n` 。如果 `n` 小于当前容量，则 `vector` 会忽略它

`resize` 和 `reserve` 都保证了 `vector` 空间的大小，至少达到它们参数所指定的大小。

#### `push_back` 与 `emplace_back` 区别

`emplace_back` 和 `push_back` 的区别，就在于底层实现的机制不同。

- `push_back` 向容器尾部添加元素时，首先会创建这个元素，然后再将这个元素拷贝或者移动到容器中（如果是拷贝的话，事后会自行销毁先前创建的这个元素）；
- `emplace_back` 是C++ 11标准新增加的成员函数。在实现时，则是直接在容器尾部创建这个元素，省去了拷贝或移动元素的过程。


## `map` 与 `unordered_map`
- `map` 内部结构是由[RBT](https://transformerswsz.github.io/2021/08/23/BST%20&%20AVL%20&%20RBT/)来实现的。查找、插入、删除都很稳定，时间复杂度为 $O(log_2 n)$
- `unordered_map` 内部是一个hash_table，一般是由一个大vector，vector元素节点可挂接链表来解决冲突。
    - `hash_map`：由于在C++标准库中没有定义散列表hash_map，标准库的不同实现者将提供一个通常名为hash_map的非标准散列表。因为这些实现不是遵循标准编写的，所以它们在功能和性能保证上都有微妙的差别。从C++11开始，哈希表实现已添加到C++标准库标准。决定对类使用备用名称，以防止与这些非标准实现的冲突，并防止在其代码中有hash_table的开发人员无意中使用新类。所选择的备用名称是unordered_map，它更具描述性，因为它暗示了类的映射接口和其元素的无序性质。可见 `hash_map` 跟 `unordered_map` 本质是一样的，只不过 `unordered_map` 被纳入了C++标准库标准。

{% asset_img unordered_map.jpeg %}

#### 适用场景
- 若考虑有序，查询速度稳定，非频繁查询那么考虑使用 `map`
- 若非常高频查询(100个元素以上，unordered_map都会比map快)，内部元素可非有序，数据大超过1k甚至几十万上百万时候就要考虑使用 `unordered_map`

___
## 参考
- [STL中vector的实现及面试问题](https://blog.csdn.net/Payshent/article/details/73835795?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.base&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.base)
- [C/C++之vector的内存管理和效率](https://blog.csdn.net/xx18030637774/article/details/82780878)
- [快速入门c++ stl](http://c.biancheng.net/view/vip_7721.html)
- [hash_map/unordered_map原理和使用整理](https://blog.csdn.net/qq_30392565/article/details/51835770)