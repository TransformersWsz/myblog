---
title: curl发送post请求存在变量转义的问题
mathjax: true
toc: true
date: 2023-03-08 00:36:48
categories:
- OS
tags:
- Linux
- curl
- POST
- bash
---

最近在使用 `curl` 发送post请求的时候，需要带上自定义的变量，示例如下：
```bash
name="Tom"
age=18
msg="my name is ${name}, age is ${age}"
echo ${msg}
curl -X POST "http://xxx.com" -H "Content-Type:application/json" -d "{\"message\":{\"header\":{\"body\":[{\"type\":\"TEXT\",\"content\":\"${msg}\"}]}}"
```

<!--more-->


有几点需要注意：
- json的key必须是双引号 `"content"` ，所以使用 `\` 进行转义
- 如果 `-d` 后面的使用单引号 `'{"message": ...}'` ，那么必须是 `"'${msg}'"` 进行转义，但是 `msg` 里不能有空格等特殊字符，所以建议采用第一点

___

## 参考
- [curl使用参数引用的方式发送POST请求](http://www.huamo.online/2017/06/17/curl%E4%BD%BF%E7%94%A8%E5%8F%82%E6%95%B0%E5%BC%95%E7%94%A8%E7%9A%84%E6%96%B9%E5%BC%8F%E5%8F%91%E9%80%81POST%E8%AF%B7%E6%B1%82/)