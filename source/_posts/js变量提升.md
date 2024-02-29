---
title: js变量提升
mathjax: true
toc: true
date: 2019-01-16 17:10:43
updated: 2019-01-16 17:10:43
categories:
- 编程语言
tags:
- JavaScript
- 变量提升
---

## 例1

```js
var a = 100;
function f() {
    console.log(a);
    if (!a) {
        var a = 200;
    }
    console.log(a);
}
f()
// undefined
// 200
```

## 例2

```js
var a = 100;
function f() {
    a = 200;
    return ;
    function a() {
        
    }
}
f();
console.log(a);
// 100
```

如果你习惯了强类型语言的编程方式，那么看到上述输出结果你肯定会大吃一惊。



# `js` 作用域

我们来看一下 `C++` 的一个例子：

```cpp
#include <iostream>

using namespace std;

int main()
{
	int x = 100;
	cout << x << endl;
	if (1)
	{
		int x = 200;
		cout << x << endl;
	}
	cout << x << endl;
	return 0;
}
// 100
// 200
// 100
```

再来看一个 `js` 的例子：

```js
var a = 100;
console.log(a);
if (true) {
    var a = 200;
    console.log(a);
}
console.log(a);
// 100
// 200
// 200
```

 `if` 代码块中的变量覆盖了全局变量。那是因为 `js` 只有<span style="color: red; font-size: 20px">全局作用域和函数作用域，没有块作用域。</span>块内的变量 `x` 影响到了全局变量 `x` 。

## `js` 实现块级作用域效果

```js
function f() {
    var x = 100;
    console.log(x);
    if (true) {
        (function() {
            var x = 200;
            console.log(x);
        }());
    }
    console.log(x);
}
// 100
// 200
// 100
```

其本质上利用了 `js` 的函数作用域来模拟实现块级作用域。

# Hoisting in `js`

在 `js` 中，变量进入一个作用域有以下方式：

- 变量定义： `var a`
- 函数形参：函数的形参存在于作用域中—— `function f(a, b) {}`

在代码运行前，函数声明和变量定义通常会被解释器移动到其所在作用域的最顶部。如下：

```js
function f() {
    test();
    var a = 100;
}
```

上面代码被解释器解释后，将会变成如下形式：

```js
function f() {
    var a;
    test();
    a = 100;
}
```

<span style="color: green; font-size: 20px"> `hoisting` 只是将变量的定义上升，但变量的赋值并不会上升。</span>

再来看一个例子：

```js
function f() {
    f1();
    f2();
    var f1 = function f1() {
        console.log("error");
    };
    function f2() {
        console.log("normal");
    }
}
f();
// TypeError: f1 is not a function
// normal
```

首先 `var f1` 会上升到函数顶部，但是此时 `f1` 为 `undefined` ，所以执行报错。但对于函数 `f2` ，函数本身也是一种变量，存在变量上升的现象，也会上升到函数顶部，所以 `f2()` 能顺利进行。

# 回顾

例1等同于如下代码：

```js
var a = 100;
function f() {
    var a;
    console.log(a);
    if (!a) {
       a = 200;
    }
    console.log(a);
}
f()
// undefined
// 200
```

例2等同于如下代码：

```js
var a = 100;
function f() {
    function a() {
    }
    a = 200;
    return ;   
}
f();
console.log(a);
// 100
```

___

## 参考
- [Javascript作用域和变量提升](https://segmentfault.com/a/1190000003114255#articleHeader1)