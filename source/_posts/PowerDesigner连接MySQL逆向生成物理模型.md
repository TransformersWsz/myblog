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


<font color=red>123423test 都算法</font> 这一点非常重要

## 下载安装ODBC
到[MySQL官网](https://dev.mysql.com/downloads/connector/odbc/)上下载ODBC，选择<font color=red>mysql-connector-odbc-5.3.9-win32.msi</font> 这一点非常重要，下面会说明理由。安装就很简单了，一路next下去

{% asset_img 1.png %}

## 配置ODBC数据源
1. 打开管理工具（不知道在哪儿的话，可以问cortana），双击<font color=red>ODBC数据源(32位)</font>，如下图所示：

{% asset_img 2.png %}

2. 点击添加，选择<font color=red>MySQL-ODBC-5.3-Unicode-Driver</font>

{% asset_img 3.png %}

<!-- 3. 点击完成，会弹出配置界面，前面两个随便填写，<font color=red>User和Password就填写你连接数据库的用户名和密码，Database选择你所要连接的数据库</font>，点击Test会弹出连接成功的提示框

{% asset_img 4.png %}，点击OK就配置完成了 -->
