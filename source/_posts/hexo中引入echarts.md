---
title: Hexo中引入Echarts
mathjax: true
toc: true
date: 2022-05-27 20:45:12
categories:
- 软件工具
tags:
- Hexo
- Echarts
---

最近发现一个插件`hexo-tag-echarts`，可以在Hexo中引入Echarts。

<!--more-->

## 插件安装

1. npm安装

```sh
npm install hexo-tag-echarts --sav
```
2. 添加cdn外链
   
在`node_modules/hexo-tag-echarts/echarts-template.html`文件中添加如下一行：
```html
<script src="https://github.com/TransformersWsz/TransformersWsz.github.io/releases/download/echarts/echarts.min.js"></script>
```

## 示例

{% echarts 400 '90%' %}
{
    tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c} ({d}%)'
    },
    legend: {
        orient: 'vertial',
        left: 10,
        data: ['直接访问', '邮件营销', '联盟广告', '视频广告', '搜索引擎']
    },
    series: [
        {
            name: '访问来源',
            type: 'pie',
            radius: ['50%', '70%'],
            avoidLabelOverlap: false,
            label: {
                show: false,
                position: 'center'
            },
            emphasis: {
                label: {
                    show: true,
                    fontSize: '30',
                    fontWeight: 'bold'
                }
            },
            labelLine: {
                show: false
            },
            data: [
                {value: 335, name: '直接访问'},
                {value: 310, name: '邮件营销'},
                {value: 234, name: '联盟广告'},
                {value: 135, name: '视频广告'},
                {value: 2000, name: '搜索引擎'}
            ]
        }
    ]
};
{% endecharts %}

## 原理