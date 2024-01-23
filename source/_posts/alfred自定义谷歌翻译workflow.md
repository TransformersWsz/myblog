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
        alfred_results.append({
            "title": translator.translate(input_text, dest=d[lang]).text,
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

___

## 参考
- [googletrans](https://py-googletrans.readthedocs.io/en/latest/)
- [Alfred工作流workflows实例](https://jlovec.net/2020/12/26/alfred-gong-zuo-liu-workflows-shi-li/)