---
title: Spring学习笔记一
mathjax: true
toc: true
date: 2022-07-18 00:26:53
categories:
- Software Engineering
tags:
- Spring
- IoC
---
公司项目中用到了Spring框架，虽然本科的时候接触过，但对其原理一知半解，现在重新学习一下。

<!--more-->

## Spring的整体框架
![spring](./Spring学习笔记一/spring.png)
其中包括四个模块：
- `Beans`：实现Bean的工厂模式，Bean可以理解为组件，是JEE中基本的代码组织单位，Spring中Bean形式是普通Java类
- `Core`：Spring框架的核心，提供控制反转/依赖注入功能
- `Context`：表示Spring应用的环境，通过此模块可访问任意Bean，ApplicationContext接口是该模块的关键组成
- `SpEL`：提供对表达式语言(SpEL)支持

如果学习的话，在 `pom.xml` 中只需引入前三个模块。

## `applicationContext.xml`
这里介绍一下各配置项的说明：

![xml](./Spring学习笔记一/xml.png)


## 最佳实践
本人在 [spl](https://github.com/TransformersWsz/spl) 建立了一个最简单的Spring项目，可以直接使用，仅供学习。

___

## 参考
- [Spring Bean定义](http://c.biancheng.net/spring/bean-definition.html)
- [Spring 框架模块](https://www.qikegu.com/docs/1468)