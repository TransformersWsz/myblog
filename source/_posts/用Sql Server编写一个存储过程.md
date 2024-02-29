---
title: 用Sql Server编写一个存储过程
mathjax: true
date: 2017-06-06 22:54:00
updated: 2017-06-06 22:54:00
categories:
- DataBase
tags:
- Sql Server
- 存储过程
---

今天数据库上机要求编写一个存储过程来体会sql server的可编程性。

<!--more-->

## 题目如下：
> 数据库中有一张表 student, 有两列分别是xh varchar(10), xm  varchar(50)，xh是主码。 现在要求编写一个存储过程，传入两个用分号分隔的字符串（如xhStr=’01;02;03;04’, xmStr=’张三;李斯;王五;赵六’, 其中字符串的长度不限，里面的分号数目也不限，由用户传入）, 存储过程完成如下功能：
把对应的两个字符串中的分号前面的字符提取，插入到student表对应的xh和xm列中。
注意：需要判断传入的字符串中分号数目是否一致，否则不让插入需要判断学号是否存在，如果存在，就不插入，而是更新姓名。

## 代码
```sql
--下面是定义函数（计算某字符在字符串中出现的次数）
create function CalcCounts
(	
	@searchstr varchar(max),
	@valuestr varchar(max)
)
returns int
as
begin
	declare @index int
	declare @count int

	set @index = charindex(@valuestr,@searchstr,0)

	set @count = 0

	while @index > 0
	begin
		set @count = @count+1
		set @searchstr = substring(@searchstr,@index+len(@valuestr),len(@searchstr))
		set @index = charindex(@valuestr,@searchstr,0)
	end
	
	return @count

end
```
___

```sql
--编写存储过程
create proc say_hello
	@xhstr varchar(max),
	@valuestr varchar(max),
	@xmstr varchar(max)
as
begin
	declare @xhindex int
	declare @xmindex int
	declare @indexcount int
	
	declare @xm_toinsert varchar(50)
	declare @subxh_front varchar(10)
	declare @subxm_front varchar(50)

	if(dbo.CalcCounts(@xhstr,@valuestr)= dbo.CalcCounts(@xmstr,@valuestr))
	begin
		print('分号一致，可以插入')
		set @indexcount = dbo.CalcCounts(@xhstr,@valuestr)

		while @indexcount >= 0
		begin

			if @indexcount = 0
			begin
				set @subxh_front = substring(@xhstr,1,len(@xhstr))
				set @subxm_front = substring(@xmstr,1,len(@xmstr))
			end

			else
			begin
				set @xhindex = charindex(@valuestr,@xhstr,1)
				set @xmindex = charindex(@valuestr,@xmstr,1)
		
				--截取xh待插入部分
				set @subxh_front = substring(@xhstr,1,@xhindex-1)
			
				--截取xm待插入部分
				set @subxm_front = substring(@xmstr,1,@xmindex-1)

				--截取字符串后面部分
				set @xhstr = substring(@xhstr,@xhindex+1,len(@xhstr))
				set @xmstr = substring(@xmstr,@xmindex+1,len(@xmstr))
			end
			
			--执行插入过程
			select @xm_toinsert = xm from student where xh = @subxh_front
			if @xm_toinsert is not null
			begin
				update student set xm = @subxm_front where xh = @subxh_front
			end

			else
			begin
				insert into student values(@subxh_front,@subxm_front)
			end

			set @indexcount = @indexcount-1
		end
	end
	
	else
	begin
		print('分号不一致，无法插入')
	end
end
```
本道编程题较为基础，算是练一下手了！