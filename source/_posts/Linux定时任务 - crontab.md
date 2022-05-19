---
title: Linux定时任务 - crontab
mathjax: true
date: 2019-02-19 21:05:00
categories:
- OS
tags:
- 定时任务
- Linux
- crontab
---

Linux系统是由 `cron` 这个系统服务来控制的。Linux系统上面原本就有非常多的计划性任务，因此这个系统服务是默认启动的。但是使用者也可以设置计划任务，Linux系统提供了控制计划任务的命令：`crontab`

<!--more-->

# `crond` 进程

`crond` 是Linux下用来周期性的执行某种任务或等待处理某些事件的一个守护进程，与windows下的计划任务类似，当安装完成操作系统后，默认会安装此服务工具，并且会自动启动 `crond` 进程，<font color="green">`crond` 进程每分钟会定期检查是否有要执行的任务，如果有要执行的任务，则自动执行该任务。</font>

# 任务调度

Linux下的任务调度分为两类：

- 系统任务调度：系统周期性所要执行的工作。

  - 常见的系统工作有：

    - 写缓存数据到硬盘
    - 日志清理等

  - 全局配置文件( `/etc` 目录 )

    - `cron.d` : 系统自动定期执行的任务。
    - `crontab` : 设定定时任务执行文件。
    - `cron.deny` : 用于控制不让哪些用户使用 `crontab` 的功能。
    - `cron.hourly` : 每小时执行一次的任务。
    - `cron.daily` : 每天执行一次的任务。
    - `cron.weekly` : 每周执行一次的任务。
    - `cron.monthly` : 每个月执行一次的任务。

  - `/etc/crontab`

    ```bash
    SHELL=/bin/bash
    PATH=/sbin:/bin:/usr/sbin:/usr/bin
    MAILTO=root
    
    # For details see man 4 crontabs
    
    # Example of job definition:
    # .---------------- minute (0 - 59)
    # |  .------------- hour (0 - 23)
    # |  |  .---------- day of month (1 - 31)
    # |  |  |  .------- month (1 - 12) OR jan,feb,mar,apr ...
    # |  |  |  |  .---- day of week (0 - 6) (Sunday=0 or 7) OR sun,mon,tue,wed,thu,fri,sat
    # |  |  |  |  |
    # *  *  *  *  * user-name  command to be executed
    ```

    - `SHELL` 指定来系统要使用哪个shell，这里是bash。
    - `PATH` 指定系统执行命令的路径。
    - `MAILTO` 指定crond的任务执行信息将通过电子邮件发送给root用户。

- 用户任务调度：用户定期要执行的工作。

  - 常见的用户工作有：
    - 数据备份
    - 定时邮件提醒等
  - 所有用户定义的 `crontab` 文件都被保存在 `/var/spool/cron` 目录中。其文件名与用户名一致。每个用户都有自己的 `cron` 配置文件,通过 `crontab -e` 就可以编辑,一般情况下我们编辑好用户的 `cron` 配置文件保存退出后,系统会自动就存放于 `/var/spool/cron/` 目录中,文件以用户名命名。Linux的 `crond` 进程是每隔一分钟去读取一次 `/var/spool/cron` , `/etc/crontab` , `/etc/cron.d` 下面所有的内容。

# `crontab` 格式说明

```bash
# .---------------- minute (0 - 59)
# |  .------------- hour (0 - 23)
# |  |  .---------- day of month (1 - 31)
# |  |  |  .------- month (1 - 12) OR jan,feb,mar,apr ...
# |  |  |  |  .---- day of week (0 - 6) (Sunday=0 or 7) OR sun,mon,tue,wed,thu,fri,sat
# |  |  |  |  |
# *  *  *  *  * user-name  command to be executed
```

- `minute` : 表示分钟，可以是从0到59之间的任何整数。

- `hour` : 表示小时，可以是从0到23之间的任何整数。

- `day` : 表示日期，可以是从1到31之间的任何整数。

- `month` : 表示月份，可以是从1到12之间的任何整数。

- `week` : 表示星期几，可以是从0到7之间的任何整数，这里的0或7代表星期日。

- `command` : 要执行的命令，可以是系统命令，也可以是自己编写的脚本文件。

在以上各个字段中，还可以使用以下特殊字符：

- `*` : 代表所有可能的值，例如month字段如果是星号，则表示在满足其它字段的制约条件后每月都执行该命令操作。

- `,` : 可以用逗号隔开的值指定一个列表范围，例如，“1,2,5,7,8,9”

- `-` : 可以用整数之间的中杠表示一个整数范围，例如“2-6”表示“2,3,4,5,6”

- `/` : 可以用正斜线指定时间的间隔频率，例如“0-23/2”表示每两小时执行一次。同时正斜线可以和星号一起使用，例如*/10，如果用在minute字段，表示每十分钟执行一次。

# 最佳实践

1. 在 `/root` 编写一个 `hello.py` 脚本，内容如下：

```python
print("Hello World")
```

2. 在 `/root` 编写一个 `test.sh` 脚本，内容如下：

```bash
#!/bin/bash
python /root/hello.py >> test.txt
```

3. 输入 `crontab -e` 命令进入编辑模式，编辑内容如下：

```bash
*/5 * * * * /root/test.sh
```

4. `crontab -l` 和 `crontab -r` 分别可以列出当前用户定时任务和删除当前用户定时任务

## 示例

1. `* * * * * commad` -> 每一分钟执行一次command
2. `3,15 8-11 */2 * * /etc/init.d/network restart` -> 每隔两天的上午8点到11店的第3和第15分钟执行 `/etc/init.d/network restart`

# 踩点

1. 有时我们创建了一个crontab，但是这个任务却无法自动执行，而手动执行这个任务却没有问题，这种情况一般是由于在crontab文件中没有配置环境变量引起的。在crontab文件中定义多个调度任务时，需要特别注意的一个问题就是环境变量的设置，因为我们手动执行某个任务时，是在当前shell环境下进行的，程序当然能找到环境变量，而系统自动执行任务调度时，是不会加载任何环境变量的，因此，就需要在crontab文件中指定任务运行所需的所有环境变量，这 样，系统执行任务调度时就没有问题了。不要假定cron知道所需要的特殊环境，它其实并不知道。所以你要保证在shell脚本中提供所有必要的路径和环境变量，除了一些自动设置的全局变量。例如上文的：<font color="color">#!/bin/bash</font>
2. 新创建的cron job，不会马上执行，至少要过2分钟才执行。如果重启cron则马上执行。
___
## 参考

- [Linux定时任务Crontab命令详解](https://www.cnblogs.com/intval/p/5763929.html)
- [Linux 定时任务crontab_014](https://www.cnblogs.com/zoulongbin/p/6187238.html)