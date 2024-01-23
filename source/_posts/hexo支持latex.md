---
title: hexo支持latex
mathjax: true
toc: true
date: 2022-03-08 00:12:04
categories:
- tools
tags:
- Hexo
- latex
---
最近在新电脑上重新搭了个博客，为了使hexo开启latex支持又踩了一次坑，在此记录一下。

<!--more-->

## 卸载旧版渲染引擎
```bash
npm uninstall hexo-renderer-marked --save
npm uninstall hexo-math --save
```

## 安装新引擎
```bash
npm install hexo-renderer-kramed --save
npm install hexo-renderer-mathjax --save
```

## 更改库文件

### `node_modules/hexo-renderer-kramed/lib/renderer.js`
将
```javascript
// Change inline math rule
function formatText(text) {
    // Fit kramed's rule: $$ + \1 + $$
    return text.replace(/`\$(.*?)\$`/g, '$$$$$1$$$$');
}
```
修改为：
```javascript
// Change inline math rule
function formatText(text) {
    return text;
}
```

### `node_modules/kramed/lib/rules/inline.js`
latex与markdown语法上有语义冲突，hexo默认的转义规则会将一些字符进行转义，所以我们需要对默认的规则进行修改。更改后为：
```javascript
var inline = {
  // escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
  escape: /^\\([`*\[\]()#$+\-.!_>])/,
  autolink: /^<([^ >]+(@|:\/)[^ >]+)>/,
  url: noop,
  html: /^<!--[\s\S]*?-->|^<(\w+(?!:\/|[^\w\s@]*@)\b)*?(?:"[^"]*"|'[^']*'|[^'">])*?>([\s\S]*?)?<\/\1>|^<(\w+(?!:\/|[^\w\s@]*@)\b)(?:"[^"]*"|'[^']*'|[^'">])*?>/,
  link: /^!?\[(inside)\]\(href\)/,
  reflink: /^!?\[(inside)\]\s*\[([^\]]*)\]/,
  nolink: /^!?\[((?:\[[^\]]*\]|[^\[\]])*)\]/,
  reffn: /^!?\[\^(inside)\]/,
  strong: /^__([\s\S]+?)__(?!_)|^\*\*([\s\S]+?)\*\*(?!\*)/,
  // em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
  em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
  code: /^(`+)\s*([\s\S]*?[^`])\s*\1(?!`)/,
  br: /^ {2,}\n(?!\s*$)/,
  del: noop,
  text: /^[\s\S]+?(?=[\\<!\[_*`$]| {2,}\n|$)/,
  math: /^\$\$\s*([\s\S]*?[^\$])\s*\$\$(?!\$)/,
};
```

### `node_modules/hexo-renderer-mathjax/mathjax.html`
将最后一行改为：
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>
```

## 开启mathjax
打开<font color="red">**主题**</font>目录下的 `_config.yml` 文件，加入如下：
```yaml
mathjax:
  enable: true
  per_page: true
```

在写博客的时候需要开启latex就加上字段说明：
```markdown
title: test
mathjax: true
```
但每次都这么加显得很麻烦，可以在 `scaffolds/post.md` 添加如下模版：
```markdown
---
title: {{ title }}
date: {{ date }}
categories: 
tags:
mathjax: true
toc: true
---
```
这样就无需每次手写了。
___

## 参考
- [Hexo博客中使用Latex](https://blog.csdn.net/weixin_44191286/article/details/102702479)