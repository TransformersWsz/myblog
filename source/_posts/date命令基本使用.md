---
title: date命令基本使用
mathjax: true
toc: true
date: 2022-12-23 00:43:10
categories:
- OS
tags:
- Linux
- date
---

在维护一些定时脚本任务的时候，经常需要使用该命令。在此做一个记录：

<!--more-->

```python
!date
```

    Fri Dec 23 00:33:25 CST 2022



```python
!date +"%Y/%m/%d %H:%M:%S"
```

    2022/12/23 00:33:57



```python
!date -d "1 year ago" +"%Y-%m-%d %H:%M:%S"
```

    2021-12-23 00:41:50



```python
!date -d "20221216 12:17:17 2 minutes ago" +"%Y-%m-%d %H:%M:%S"
```

    2022-12-16 12:15:17



```python
!date -d "20221216 12:17:17 2 minutes" +"%Y-%m-%d %H:%M:%S"
```

    2022-12-16 12:19:17

