---
title: top命令
mathjax: true
toc: true
date: 2024-01-23 17:50:41
categories:
- OS
tags:
- Linux
- top
---

![top](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/04eb5801268567e6bf9c714fc114282d7d3d36f8/image.4kkhwptk8tg0.png)

<!--more-->

## 字段介绍

- 时间：
    - 当前时间
    - 系统运行时间
    - 当前登录用户数
    - 系统负载，即任务队列的平均长度。三个数值分别为 1分钟、5分钟、15分钟前到现在的平均值
    
- 任务：
    - 进程总数
    - 正在运行的进程数
    - 睡眠的进程数
    - 停止的进程数
    - 僵尸进程数
    
- CPU：
    - us用户空间占用CPU百分比
    - sy内核空间占用CPU百分比
    - ni用户进程空间内改变过优先级的进程占用CPU百分比
    - id空闲CPU百分比
    - wa 等待输入输出的CPU时间百分比
    - hi硬件CPU中断占用百分比
    - si软中断占用百分比
    - st虚拟机占用百分比
    
- 内存
    - 物理内存总量
    - 使用的物理内存总量
    - 空闲内存总量
    - 用作内核缓存的内存量

- 交换区
    - 交换区总量
    - 使用的交换区总量
    - 空闲交换区总量
    - 缓冲的交换区总量,内存中的内容被换出到交换区，而后又被换入到内存，但使用过的交换区尚未被覆盖，该数值即为这些内容已存在于内存中的交换区的大小,相应的内存再次被换出时可不必再对交换区写入

## 指定字段排序查看

- 按照内存占用大小倒序：Shift+M
- 按照CPU占用大小倒序：Shift+P

___

## 参考
- [linux的top命令参数详解](https://www.cnblogs.com/ggjucheng/archive/2012/01/08/2316399.html)