---
title: Oracle中call和exec区别
mathjax: true
toc: true
date: 2017-10-09 21:27:05
categories:
- DataBase
tags:
- Oracle
---

在<font color="red">SQL Plus</font>中这两种方法都可以使用：

- `exec pro_name(参数1..)`
- `call pro_name(参数1..)`

<!--more-->

## 区别
1. exec是SQL Plus命令，只能在SQL Plus中使用；call为SQL命令，没有限制.
2. <font color="red">存储过程或函数没有参数时,exec可以直接跟过程名（可以省略()），但call则必须带上().</font>

## 示例
### exec
- 调用存储过程
    - 有参数：`exec mypro(12,'fsdf');`
    - 没有参数：`exec mypro;`，也可以写成`exec mypro();`
- 调用函数
    - 有参数：`var counts number;exec :counts:=myfunc('fsde');`
    - 没有参数：`var counts number;exec :counts:=myfunc;`，也可以写成`var counts number;exec :counts:=myfunc();`

### call
- 调用存储过程
	- 有参数：`call mypro(23,'fth');`
	- 无参数：`call mypro();`
- 调用函数
	- 有参数：`var counts number;call myfunc('asd') into :counts;`
	- 无参数：`var counts number;call myfunc() into :counts;`

## 其他注意事项
- oracle 中字符串应该是单引号而不是双引号
- 每写完一条sql语句应加上 <font color="red">;</font>
- 为了防止call 和 exec 无参数的存储过程或函数的错误，建议全部加上<font color="red">()</font>