---
title: C++学习笔记之多态
mathjax: true
date: 2021-06-01 00:15:10
categories:
- 编程语言
tags:
- C++
---


> 多态性：相同对象收到不同消息或不同对象收到相同消息时产生不用的实现动作。

<!--more-->

多态有两种类型：

- 编译时多态性（静态多态）：通过重载函数实现。
- 运行时多态性（动态多态）：通过虚函数实现。

## 多态与非多态

实质区别就是函数地址是早绑定还是晚绑定。

- 如果函数的调用，在编译期间就可以确定函数的调用地址，并生产代码，是静态的，就是说地址是早绑定的。
- 如果函数调用的地址不能在编译期间确定，需要在运行时才确定，这就属于晚绑定。

## 多态的目的

- 封装：代码模块化。继承：可以扩展已存在的代码。两者目的都是为了代码重用。
- 多态：接口重用。不论传递过来的究竟是类的哪个对象，函数都能够通过同一个接口调用到适应各自对象的实现方法。

## 使用场景

声明基类类型的指针，利用该指针指向任意一个子类对象，调用相应的虚函数，可以根据指向的子类的不同而实现不同的方法。如果没有使用虚函数的话，即没有利用C++多态性，则利用基类指针调用相应的函数的时候，将总被限制在基类函数本身，而无法调用到子类中被重写过的函数。因为没有多态性，函数调用的地址将是固定的，因此将始终调用到同一个函数，这就无法实现“一个接口，多种方法”的目的了。

## 最佳实践

```c++
#include <iostream> 
using namespace std;

class Shape {
protected:
    int width;
    int height;
public:
    Shape(int a = 0, int b = 0)
    {
        width = a;
        height = b;
    }
    virtual void area()
    {
        cout << "Parent class area : " << -1 << endl;
    }
};

class Rectangle : public Shape {
public:
    Rectangle(int a = 0, int b = 0) : Shape(a,b) {}

    void area()
    {
        cout << "Rectangle class area: " << width * height << endl;
    }
};

class Triangle : public Shape {
public:
    Triangle(int a = 0, int b = 0) : Shape(a, b) {}

    void area()
    {
        cout << "Triangle class area: " << width * height / 2 << endl;
    }
};

// 程序的主函数
int main()
{
    Shape* shape;
    Rectangle rec(10, 7);
    Triangle  tri(10, 5);

    // 存储矩形的地址
    shape = &rec;
    // 调用矩形的求面积函数 area
    shape->area();

    // 存储三角形的地址
    shape = &tri;
    // 调用三角形的求面积函数 area
    shape->area();

    return 0;
}
```

___

## 参考

- [C++ 多态 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/37340242)
- [C++ 多态 | 菜鸟教程 (runoob.com)](https://www.runoob.com/cplusplus/cpp-polymorphism.html)
