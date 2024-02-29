---
title: args & kwargs
mathjax: true
date: 2017-12-29 21:24:18
updated: 2017-12-29 21:24:18
categories:
- 编程语言
tags:
- python
- 可变参数
---

`args` 和 `kwargs` 是python函数中最常用的传参形式，用法非常灵活。

<!--more-->

## 基本概念
Python支持可变参数，最简单的方法莫过于使用默认参数。
```python
def test_defargs(one, two=2):    # 参数one没有默认值，two的默认值为2
   print('Required argument: ', one)
   print('Optional argument: ', two)

test_defargs(1)
'''
Required argument: 1
Optional argument: 2
'''

test_defargs(1, 3)
'''
Required argument: 1
Optional argument: 3
'''
```
另外一种达到可变参数 (Variable Argument) 的方法：
使用 `*args` 和 `**kwargs` 语法。
`*args` 是可变的位置参数(positional arguments)列表，
`**kwargs` 是可变的关键词参数(keyword arguments)列表。
并且规定位置参数必须位于关键词参数之前，即 `*args` 必须位于 `**kwargs` 之前。

## 位置参数
```python
def print_hello(name, sex):
    sex_dict = {0: '先生', 1: '女士'}    # key: value
    print('hello %s %s, welcome to Python world!' %(name, sex_dict.get(sex%2, '先生')))    # if no such a key, print '先生'

print_hello('Chen', 2)    # 位置参数要求先后顺序，对应name和sex
print_hello('Chen', 3)    # 两个参数的顺序必须一一对应，且少一个参数都不可以
'''
hello Chen 女士, welcome to Python world!
hello Chen 先生, welcome to Python world!
'''
```

## 关键字参数
用于函数调用，通过“键-值”形式加以指定。
使用关键词参数可以让函数更加清晰、容易使用，同时也清除了参数的顺序需求。

以下是用关键字参数正确调用函数的实例
```python
print_hello('Chen', sex=1)    # 有位置参数时，位置参数必须在关键字参数的前面
print_hello(name='Chen', sex=1)    # 关键字参数之间不存在先后顺序的
print_hello(sex=1, name='Chen')
```
以下是错误的调用方式:
```python
print_hello(name='Chen', 1)    # 有位置参数时，位置参数必须在关键字参数的前面
print_hello(sex=1, 'Chen')
```

## 默认参数
使用关键词参数时，可为参数提供默认值，调用函数时可传可不传该默认参数的值（注意：所有位置参数必须出现在默认参数前，包括函数定义和调用）

正确的默认参数定义方式 --> 位置参数在前，默认参数在后:
```python
def print_hello2(name, sex=1):    # 默认sex=1
    sex_dict = {1: '先生', 2: '女士'}    # key: value
    print('hello %s %s, welcome to circus!' %(name, sex_dict.get(sex, '先生')))

# 错误的定义方式
# def print_hello2(sex=1, name):

# 调用时不传sex的值，则使用默认值1
print_hello2('Chen')
'''
hello Chen 先生, welcome to circus!
'''

# 调用时传入sex的值，并指定为2。无视原来的sex=1，仅本次生效，不会改变函数sex的默认值
print_hello2('Liu', sex=2)
'''
hello Liu 女士, welcome to circus!
'''
```

## 收集参数(Packing)
> 被收集到一起的位置参数或关键词参数

有些时候，我们在定义参数时不确定会传递多少个参数（或者不传递），甚至我们想要根据实际情况传入任意个参数，这个时候 `packing` 就派上了用场。
```python
def print_args(*args):
    print(args)

print_args(1)
print_args(1, 2, 3)
'''
(1,)
(1, 2, 3)
'''
```
可以看到，结果是作为一个元组(tuple)打印出来的，因为仅传入1时，输出的括号里面有一个逗号。实际上，`*args` 前的 `*` 可以理解为将我们需要传入的所有值放置在了同一个元组里面，这就是收集 (packing) 的过程。

### 位置参数的收集传递 ———— *packing
```python
def print_args2(num, *args):
    print(num)
    print(args)

print_args2(1)
print_args2(1, 2, 3)
'''
1    print(num)
()    print(args) --> 一个空元组
1    print(num)
(2, 3)    print(args) --> 剩余的参数组成的元组！
'''
```
那么这么说来 `*` 的意思就是“把剩余的所有参数收集 packing 起来”！

### 关键词参数的收集传递 ———— **packing
```python
def print_kargs(**kargs):
    print(kargs)

print_kargs(x=1, y=2, z=3)
'''
{'z': 3, 'y': 2, 'x': 1}
'''
```
结果输出了一个字典(dic)，双星号 `**` 收集了所有关键词参数。

### 分割参数
`*` 和 `**` 也可以在函数调用过程中使用，称之为分割 (unpacking)。
#### 在传递元组时，让元组的每一个元素对应一个位置参数
```python
def print_hello(name, sex):
    print(name, sex)

args = ('Chen', '男')
print_hello(*args)
'''
Chen 男
'''
```
#### 在传递词典字典时，让词典的每个键值对作为一个关键字参数传递给函数
```python
def print_info(**kargs):
    print(kargs)

kargs = {'name': 'Chen', 'sex': '男', 'time': '13'}
print_info(**kargs)
'''
{'name': 'Chen', 'sex':'男','time':'13'}
'''
```

## 位置参数、关键词参数（默认参数）、收集参数的混合使用
基本原则是：先位置参数，默认参数，收集位置参数，收集关键字参数(定义和调用都应遵循)
```python
def func(name, age, sex=1, *args, **kargs):
    print(name, age, sex, args, kargs)

func('Chen', 25, 2, 'Geo', 'SR', level=2)
'''
Chen 25 2 ('Geo', 'SR') {'level': 2}
'''
```
总结一下就是，星号 `*` 只有在定义需要使用不定数目的参数的函数，或者调用“分割”字典和序列时才有必要添加。

## 各类参数混合使用
### EXAMPLE 1
#### Tell a story
```python
def story1(**kwargs):
    return 'Once upon a time, there was a %(job)s called %(name)s.' % kwargs

def story2(**kwargs):
    return 'In the year of our lord %(year)d, there once lived a %(job)s of a royal line.' % kwargs

print(story1(name='Robin', job='brave knight'))
print(story1(job='king', name='Charlie'))
'''
Once upon a time, there was a brave knight called Robin.
Once upon a time, there was a king called Charlie.
'''

kwargs = {'name': 'Abel', 'job': 'bard'}
print(story1(**kwargs))
'''
Once upon a time, there was a bard called Abel.
'''

kwargs.pop('job')    # delete key['job'] --> kwargs = {'name': 'Abel',}
print(story1(job='witch', **kwargs))    # redefine key['job']
'''
Once upon a time, there was a witch called Abel.
'''

print(story2(year=1239, job='prince'))
print(story2(job='princess', year=1239))
'''
In the year of our lord 1239, there once lived a prince of a royal line.
In the year of our lord 1239, there once lived a princess of a royal line.
'''
```

### EXAMPLE 2
#### x to the yth power
```python
def power(x, y, *args):
    if args:
        print('Received redundant parameters:', args)
    return pow(x, y)    

print(power(2, 3))    # 2 to the 3rd power
print(power(3, 2))    # 3 to the 2nd power
print(power(x=2, y=3))    # 2 to the 3rd power
print(power(y=3, x=2))    # no order for keyword arguments
'''
8
9
8
8
'''

args = (5,) * 2    # '(5,) * 2': copy elements in the tuple 2 times. The outcome is (5, 5)
print(power(*args))    # 5 to the 5th power
'''
3125
'''

print(power(2, 5, 'Keep on trying'))
'''
Received redundant parameters: ('Keep on trying',)
32    # 2 to the 5th power
'''
```

### EXAMPLE 3
#### Bulid a function to realize range() for step>0
```python
def interval(start=0, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    result = []
    i = start
    while i < stop:
        result.append(i)
        i += step
    return result

print(interval(10))
'''
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
'''

print(interval(1, 6))
'''
[1, 2, 3, 4, 5]
'''

print(interval(3, 15, 3))
[3, 6, 9, 12]
```
___
## POSTSCRIPT
在Python 3.X当中加入了"部分剩余参数"的概念。举例如下 : 
```python
a,\*b,c = range(5)
```
中间的 *b 实际上是收集到了3个参数。

下面这个例子更加直观 : 
```python
def func1(x, y, *args, z=1):    # z=1 在函数参数中最后定义
    print(x, y, args, z)


func1(1, 2, 3, 4, 5)
'''
1 2 (3, 4, 5) 1
'''
```
Python 3.X 当中，默认参数 z=1 在函数参数中最后定义，Python 就会知道除了 传给 x 和 y, 以及 z=1 的部分，其他的剩余参数 (3, 4, 5) 都是传入到 *args 当中。

<font color="red">但在Python 2.X 并没有这一特性，所以 Python 2.X 的 z=1 必须在args和*kwargs这种剩余参数收集之前。</font>

默认参数 z=1 不是在函数参数中最后定义时，情况又是怎样？
```python
def func2(x, y, z=1, *args):
    print(x, y, z, args)


func2(1, 2, 3, 4, 5)
'''
1 2 3 (4, 5)
'''
```
3传入到 z ，而 args=(4, 5)
___
## 参考 
- [Python中的 *args 和 **kwargs](https://www.jianshu.com/p/61507f60fa29)
