---
title: shell并发编程
mathjax: true
toc: true
date: 2022-11-15 01:20:17
updated: 2022-11-15 01:20:17
categories:
- OS
tags:
- Linux
---
实际上 `&` 符号就表示将程序放入后台执行，从而实现多个程序并行。但由于机器资源有限，我们需要控制并发数量。下面是解决方案：

<!--more-->

```python
import sys

if __name__ == "__main__":
    id = sys.argv[1]
    print("Processing task {}".format(id))

```

```bash
#!/bin/bash
# 最大并发数
Thread_num=5
# 命名管道文件
Tmp_fifo=/tmp/$$.fifo

# 创建命名管道文件
mkfifo $Tmp_fifo
# 用文件句柄(命名随意)打开管道文件
exec 1000<>$Tmp_fifo
rm -f $Tmp_fifo

# 控制并发数
for i in `seq $Thread_num`
do
        # 向管道中放入最大并发数个行，供下面read读取
        echo "" >&1000
done

for i in {1..17}
do
        # 通过文件句柄读取行，当行取尽时，停止下一步（并发）
        read -u 1000
        {
                # 业务代码
                python main.py ${i}
                sleep 10s
        # 一个并发执行后要想管道中在加入一个空行，供下次使用
        echo "" >&1000
        }&
done
wait
echo "END"
```
有 `17` 个任务，控制并发数量为 `5`。`wait` 表示等待所有后台进程结束，否则的话会出现如下情况：
```
Processing task 1
Processing task 5
Processing task 2
Processing task 4
Processing task 3
Processing task 7
Processing task 6
Processing task 8
Processing task 10
Processing task 9
Processing task 15
Processing task 14
Processing task 12
Processing task 13
Processing task 11
END
Processing task 16
Processing task 17
```

___

## 参考

- [shell脚本中多线程和控制并发数量](https://blog.csdn.net/weixin_42170236/article/details/117821258)

