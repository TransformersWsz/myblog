---
title: java8 Stream流常用示例
mathjax: true
toc: true
date: 2022-09-21 23:35:28
categories:
- 编程语言
tags:
- Java
- Stream
---
Java8推出了Stream Api，开发者能够以声明的方式来流式处理数据。Stream可以让臃肿的代码变得更加简洁、高效。

> Stream将要处理的元素集合看作一种流， 流在管道中传输，并且可以在管道的节点上进行处理，比如筛选、排序、聚合等。元素流在管道中经过中间操作的处理，最后由最终操作得到前面处理的结果。

<!--more-->

下面列举了Api的常用示例：
```java
package org.example;

import org.apache.commons.lang.StringUtils;

import java.util.Arrays;
import java.util.List;
import java.util.Comparator;
import java.util.stream.Collectors;

/**
 * @Author: swift
 * @CreateAt: 2022/09/16 17:11
 * @Description: Stream 示例
 */
public class Temp {
    public static void main(String[] args) {
        System.out.println("stream learning");

        String[] arr = {"123", "34", "23", "234"};
        // 遍历数组
        Arrays.asList(arr).forEach(System.out::println);
        // 将字符串数组转换为整数数组
        List<Integer> intList = Arrays.asList(arr)
                                .stream()
                                .map(ele -> Integer.parseInt(ele))
                                .collect(Collectors.toList());

        // 获取字符串数组中最大值：234
        Integer maxValue = Arrays.asList(arr)
                            .stream()
                            .map(ele -> Integer.parseInt(ele))
                            .max(Comparator.comparing(Integer::intValue))
                            .get();

        // 或者简写如下(求最小值)：23
        Arrays.asList(arr)
                .stream()
                .mapToInt(ele->Integer.parseInt(ele))
                .min()
                .ifPresent(minValue -> System.out.println("Min: " + minValue));

// 复杂的对象求最大值
//        users.stream()
//                .max(Comparator.comparing(u -> u.getUserName()))
//                .ifPresent(e -> System.out.println("Max: " + e.getUserName()));

        // 过滤非数字字符串，并将剩下的数字字符串排序然后输出：12，24，45，231，2342
        String[] data = {"231", "", "as", "12", "3453", "asd", "24", "2342", "45"};
        Arrays.asList(data)
                .stream()
                .filter(ele -> !ele.isEmpty() && StringUtils.isNumeric(ele))
                .map(ele -> Integer.parseInt(ele))
                .sorted()
                .limit(5)
                .forEach(System.out::println);
    }
}
```

___

## 参考

- [Java 8 Stream](https://www.runoob.com/java/java8-streams.html)