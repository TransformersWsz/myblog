---
title: PowerDesigner连接MySQL逆向生成物理模型
mathjax: true
date: 2017-09-17 22:40:33
updated: 2017-09-17 22:40:33
categories:
- tools
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



## 配置ODBC数据源
1.打开管理工具（不知道在哪儿的话，可以问cortana），双击<font color=red>ODBC数据源(32位)</font>，如下图所示：

{% asset_img 2.png %}

2.点击添加，选择<font color=red>MySQL ODBC 5.3 Unicode Driver</font>

{% asset_img 3.png %}

3.点击完成，会弹出配置界面，前面两个随便填写，<font color=red>User和Password就填写你连接数据库的用户名和密码，Database选择你所要连接的数据库</font>，点击Test会弹出连接成功的提示框

{% asset_img 4.png %}，点击OK就配置完成了

## 使用PowerDesigner逆向生成物理模型
1.打开PowerDesigner新建模型，DBMS选择MySQL5.0

{% asset_img 5.png %}

2.菜单栏 Database -> Connect，点击弹出连接界面。从下拉菜单中选择刚刚配置的ODBC数据源，点击Connect即可连接成功。

{% asset_img 6.png %}

<font color=red>注意：现在来解释一下为什么选择32位的安装包，如果选择了64位的，此时点击Connect会弹出报错框：在指定的DSN中，驱动程序和应用程序的体系结构不匹配，SQLSTATE=IM014.具体原因我也不知道。</font>

3.菜单栏 Database -> Update Model From Database...，弹出如下界面：

{% asset_img 7.png %}

点击确定，PowerDesigner默认选中所有数据库的所有表，要想生成我们想要的数据库的物理模型，先反选一下Deselect All，

{% asset_img 8.png %}

再选中partysystem数据库，Select All即选中该数据库中的所有表

{% asset_img 9.png %}

最后点击OK，即生成我们想要的物理模型。

{% asset_img 10.png %}

我这个数据库里面的表结构比较单一，所以生成的物理模型很简单。
