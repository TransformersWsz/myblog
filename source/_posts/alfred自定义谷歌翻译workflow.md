---
title: alfred自定义谷歌翻译workflow
mathjax: true
toc: true
date: 2024-01-23 13:05:05
categories:
- tools
tags:
- Alfred
- Workflow
---
如果要实现自定义workflow，则必须安装付费版的alfred，囊中羞涩的话可以自行淘宝。自定义步骤如下：

<!--more-->

## 1. 新建空的workflow，填写基本信息

![blank workflow](https://cdn.jsdelivr.net/gh/TransformersWsz/picx-images-hosting@master/image.2bpwgci5ntes.webp)

## 2. 开发python脚本
打开该workflow所在目录，进行下面步骤：

1. 首先安装谷歌翻译库：
```bash
pip install googletrans==3.1.0a0
```
2. 编写py脚本
```python
import sys
import json
from googletrans import Translator

def main(input_text):
    d = {
        "en": "zh-CN",
        "zh-CN": "en"
    }
    translator = Translator()
    lang = translator.detect(input_text).lang
    alfred_results = []
    # 中英文互译
    if lang in d:
        text = translator.translate(input_text, dest=d[lang]).text
        alfred_results.append({
            "title": text,
            "arg": text,    # 该参数不可省略，将用于后续的剪贴板复制；否则后续动作无法触发
            "icon": {
                "path": "./google_translate.png"
            }
        })
    else:
        alfred_results.append({
            "title": "未识别语种",
            "icon": {
                "path": "./google_translate.png"
            }
        })
    return json.dumps({
        "items": alfred_results
    }, ensure_ascii=False)


if __name__ == "__main__":
    resp = "no input text to translate"
    if len(sys.argv) >= 2:
        input_text = "\t".join(sys.argv[1:])
        resp = main(input_text)
    sys.stdout.write(resp)
```

## 3. 编辑工作流
1. 新建script filter：

![new scipt filter](https://cdn.jsdelivr.net/gh/TransformersWsz/picx-images-hosting@master/image.4s7zzod45jc0.webp)

配置信息说明:
- 触发谷歌翻译关键词：`tr`
- 将输入看做`{query}`
- 调用python脚本进行翻译：`python ./translate.py "{query}"`
- 避免一些转义符

![config](https://cdn.jsdelivr.net/gh/TransformersWsz/picx-images-hosting@master/image.2bw7bcg882m8.webp)

2. 新增剪贴板

在filter后面接一个clipboard：

![clipboard](https://cdn.jsdelivr.net/gh/TransformersWsz/picx-images-hosting@master/image.723f5xy7w2o0.webp)

## 4. 调试工作流

右侧有个虫子标记，点击。然后调起alfred，输入命令测试，下面的控制台会打印日志信息：

![debug](https://cdn.jsdelivr.net/gh/TransformersWsz/picx-images-hosting@master/image.5cwhhyuhgmg0.webp)

如果上述步骤一切顺利的话，你的工作流就实现了。
___

## 参考
- [googletrans](https://py-googletrans.readthedocs.io/en/latest/)
- [Alfred工作流workflows实例](https://jlovec.net/2020/12/26/alfred-gong-zuo-liu-workflows-shi-li/)