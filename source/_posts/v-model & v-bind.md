---
title: v-model & v-bind
mathjax: true
date: 2018-05-08 22:24:56
updated: 2018-05-08 22:24:56
categories:
- Software Engineering
tags:
- Vue
- 前端
- JavaScript
---
记录一下两种指令的用法。

<!--more-->

## v-model

我们可以使用 `v-model` 指令在 `<input>` (`<input>` 标签有多种类型，如 `button、select` 等等)及 `<textarea>` 元素上进行双向数据绑定。但 `v-model` 本质上不过是语法糖。它负责监听用户的输入事件以更新数据，并对一些极端场景进行一些特殊处理。

`v-model` 会忽略所有表单元素的 `value`、`checked`、`selected` 特性的初始值而总是将 Vue 实例的数据作为数据来源。你应该通过 JavaScript 在组件的 `data`选项中声明初始值：

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <script src="https://cdn.jsdelivr.net/npm/vue@2.5.16/dist/vue.js"></script>
    <title>vue</title>
</head>
<body>
    <div id="app">
        <input v-model="message">
        <p>The input value is : {{message}}</p>
    </div>
    <script>
        var app = new Vue({
            el: '#app',
            data: {
                message: 'Hello Word!'
            }
        })
    </script>
</body>
</html>
```

那么输入框的初始值就是 <font color="green">Hello World!</font> 。



实际上，由于`v-model` 只是语法糖， `<input v-model="message">` 与下面的两行代码是一致的：

```html
<input v-bind:value="message" v-on:input="message = $event.target.value" />
<input :value="message" @input="message = $event.target.value" />
```



## v-bind

它的用法是后面加冒号，跟上html元素的属性，例如：

```html
<p v-bind:class="someclass"></p>
```

如果不加 `v-bind` 那么 `someclass` 就是个常量，没有任何动态数据参与。当加上 `v-bind` 之后，它的值 `someclass` 不是字符串，而是vue实例对应的 `data.someclass` 这个变量。具体传入变量类型可参考 [Class与Style绑定](https://cn.vuejs.org/v2/guide/class-and-style.html) 。这非常适合用在通过css来实现动画效果的场合。除了class，其他大部分html原始的属性都可以通过这种方式来绑定，而且为了方便，它可以直接缩写成冒号形式，例如：

```javascript
var app = Vue({  
    el: '#app',  
    template: '<img :src="remoteimgurl">',  
    data: {    src: '',  },  
    beforeMount() {    fetch(...).then(...).then(res => this.src = res.remoteimgurl) },
})
```

上面这段代码中，默认情况下 `data.src` 是空字符串，也就说不会有图片显示出来，但是当从远端获取到图片地址之后，更新了 `data.src`，图片就会显示出来了。



## v-bind与v-model区别

有一些情况我们需要 `v-bind` 和 `v-model` 一起使用：

```html
<input :value="name" v-model="body">
```

`data.name` 和 `data.body`，到底谁跟着谁变呢？甚至，它们会不会产生冲突呢？

实际上它们的关系和上面的阐述是一样的，`v-bind` 产生的效果不含有双向绑定，所以 `:value` 的效果就是让 input的value属性值等于 `data.name` 的值，而 `v-model` 的效果是使input和 `data.body` 建立双向绑定，因此首先 `data.body` 的值会给input的value属性，其次，当input中输入的值发生变化的时候，`data.body` 还会跟着改变。

上文提到过下面两句是等价的：

```html
<input v-model="message">
<input v-bind:value="message" v-on:input="message = $event.target.value" />
```

那么 `v-model` 其实就是 `v-bind` 和 `v-on` 的语法糖。
