---
title: Swift中?和!使用总结
mathjax: true
toc: true
date: 2017-12-28 23:16:57
categories:
- 编程语言
tags:
- Swift
- 可选类型
---
## Optional(可选类型)
Swift的可选(Optional)类型，用于处理值缺失的情况。可选表示<font color="red">“那儿有一个值，并且它等于x”或者“那儿没有值，为nil”</font>。它的定义通过在类型声明后加一个 `?` 操作符来完成的:
```swift
var str = String?
```
`Optional` 其实是个 `enum` ，里面有 `None` 和 `Some` 两种类型。其实所谓的 `nil` 就是 `Optional.None` ，当你声明一个可选变量的时候没有提供初始值，它的值默认为 `nil` 。 `非nil` 就是 `Optional.Some` ，然后会通过 `Some(T)` 包装(`wrap`)原始值，这也是为什么在使用 `Optional` 的时候要拆包(`unwrap` : 从 `enum` 中取出来原始值)的原因。下面是 `enum Optional`  的定义:

```swift
public enum Optional<Wrapped> : ExpressibleByNilLiteral {

    case none

    case some(Wrapped)

    public init(_ some: Wrapped)

    public func map<U>(_ transform: (Wrapped) throws -> U) rethrows -> U?

    public func flatMap<U>(_ transform: (Wrapped) throws -> U?) rethrows -> U?

    public init(nilLiteral: ())

    public var unsafelyUnwrapped: Wrapped { get }
}
```
既然这样， 那我们如何理解上述变量的声明呢？
```swift
var str = String?
//我声明了一个Optional类型的变量，它可能包含一个String值，也可能什么都不包含，即nil
```
<font color="red">也就是说我们实际上声明的是一个 `Optional` 类型，而不是 `String` 类型。</font>

## ? 和 ! 的比较
### 举例 :
```swift
import Cocoa
var str : String?
str = "Hello World"
if str != nil{
    //print(str)
    print(str!)
}
else {
    print("字符串为nil")
}
```
输出结果为: <font color="green">Hello World</font>

#### 注意
- <font color="red">如果是执行 `print(str)` 这句话，那么输出为 `Optional("Hello World")`。</font>
- 使用 `!` 来获取一个不存在的可选值会导致运行时错误。使用 `!` 来强制解析值之前，一定要确定可选包含一个 `非nil` 的值。

怎么使用 `Optional` 值呢？在苹果文档中也有提到说，在使用 `Optional` 值的时候需要在具体的操作，比如调用方法、属性、下标索引等前面需要加上一个?，如果是 `nil` 值，也就是 `Optional.None` ，会跳过后面的操作不执行，如果有值，就是 `Optional.Some` ，可能就会拆包(`unwrap`)，然后对拆包后的值执行后面的操作，来保证执行这个操作的安全性。

举例如下：
```swift
let length = str?.count
//如果你确定有值的话，也可以这样写
//let length = str!.count
```



### 拆包(unwrap)
上文提到 `Optional` 值需要拆包才能得到原来值，并判断其值是否为空才能对其操作。下面介绍两种拆包方法：

1. 可选绑定(optional binding)

使用可选绑定（optional binding）来判断可选类型是否包含值，如果包含就把值赋给一个临时常量或者变量。可选绑定可以用在if和while语句中来对可选类型的值进行判断并把值赋给一个常量或者变量。
举例如下：
```swift
import Cocoa

var str : String? = "Hello"
let world = "World"
if let const = str{
    print(const + " " + world)
}
else {
    print("str is nil")
}
```
2. 硬解包
即直接在可选类型后面加一个 `!` 来表示它肯定有值。
举例如下：
```swift
import Cocoa

var str : String? = "Hello"
let world = "World"
if str != nil{
    print(str! + " " + world)
}
else {
    print("str is nil")
}
```

#### <font color="red">错误示范</font>
```swift
import Cocoa

var str:String?
let world = "Hi"
print(str! + world)
```
以上代码在编译阶段不会报错.因为使用了硬解包, 编译器认为可选类型是有值的, 所以编译是通过的。当代码运行起来时，会报错：
<font color="red">Fatal error: Unexpectedly found nil while unwrapping an Optional value.</font>

### 隐式拆包(自动解析)
你可以在声明可选变量时使用感叹号 `!` 替换问号`?`。这样可选变量在使用时就不需要再加一个感叹号 `!` 来获取值，它会自动解析。
举例如下:
```swift
import Cocoa

var str:String!
str = "Hello World!"

if str != nil {
   print(str)
}else{
   print("str is nil")
}
```
等于说你每次对这种类型的值操作时，都会自动在操作前补上一个 `!` 进行拆包，然后在执行后面的操作，当然如果该值是 `nil` ，会报错crash掉。

总而言之，`?` 和 `!` 坑还是很多的，需要不断在实践中检验和体会。
___

## 参考
- [Swift中 ！和 ？的区别及使用](https://www.jianshu.com/p/89a2afb82488)
- [Swift 可选(Optionals)类型](http://www.runoob.com/swift/swift-optionals.html)
