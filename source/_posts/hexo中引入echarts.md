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
`index.js`文件内容如下：
```js
var fs = require('fs'),
  path = require('path'),
  _ = require('underscore');

var filePath = path.join(__dirname, 'echarts-template.html');

function echartsMaps(args, content) {

  var template = fs.readFileSync(filePath).toString(),
    options = {};

  if (content.length) {
    var options = content;
  }

  // Output into 
  return _.template(template)({
    id: 'echarts' + ((Math.random() * 9999) | 0),
    option: options,
    height: args[0] || 400,
    width: args[1] || '81%'
  });
}


hexo.extend.tag.register('echarts', echartsMaps, {
  async: true,
  ends: true
});
```
- `args` 接收参数 `{% echarts 400 '90%' %}`
- `content` 接收具体的数据信息

详细的数据流是这样的：

`index.js`拿到markdown里的参数和数据，进行一些处理，然后将对应的字段渲染到`echarts-template.html`中。