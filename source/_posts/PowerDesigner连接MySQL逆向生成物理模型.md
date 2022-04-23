---
title: PowerDesigner连接MySQL逆向生成物理模型
mathjax: true
toc: true
date: 2017-09-17 22:40:33
categories:
- 软件工具
tags:
- MySQL
- PowerDesigner
---
根据MySQL现有的表格结构来反向生成含有依赖关系表格模型。

<!--more-->

系统环境：Win10 64位系统

## 下载安装ODBC
到[MySQL官网](https://dev.mysql.com/downloads/connector/odbc/)上下载ODBC，选择<font color=red>mysql-connector-odbc-5.3.9-win32.msi</font> 这一点非常重要，下面会说明理由。安装就很简单了，一路next下去

{% asset_img 1.png %}
