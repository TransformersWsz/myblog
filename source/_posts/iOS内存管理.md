---
title: iOS内存管理
mathjax: true
toc: true
date: 2017-08-21 20:40:45
categories:
- OS
tags:
- iOS
- Objective-C
- 内存管理
---

现在iOS开发已经是ARC的时代，但是内存管理仍是一个重点关注的问题。它是程序设计中很重要的一部分。程序在运行的过程中消耗内存，运行结束后释放占用的内存。如果程序运行时一直分配内存而不及时释放无用的内存，程序占用的内存会越来越大，直至内存消耗殆尽，程序因无内存可用导致崩溃，这就是所谓的内存泄漏。本文将介绍ObjC的内存管理方式。

<!--more-->

## 引用计数
ObjC采用引用计数来进行内存管理:

1. 每个对象都有一个关联的整数，称为引用计数器
2. 当代码需要使用该对象时，则将对象的引用计数加1
3. 当代码结束使用该对象时，则将对象的引用计数减1
4. 当引用计数的值变为0时，表示对象没有被任何代码使用，此时对象占用的内存将被释放

与之对应的消息发送方法如下:

1. 当对象被创建(`alloc 、new 、copy` 等方法)时，其引用计数初始值为1
2. 给对象发送 `retain` 消息，其引用计数加1
3. 给对象发送 `release` 消息，其引用计数减1
4. 当对象引用计数归0时，ObjC给对象发送 `dealloc` 消息销毁对象

下面通过一个例子来说明(关闭Xcode的 Automatic Reference Counting 功能):

场景: 有一个宠物中心（内存），可以派出小动物（对象）陪小朋友们玩耍（对象引用者），现在xiaoming想和小狗一起玩耍。

新建Dog类，重写其创建和销毁的方法

```ObjectiveC
#import "Dog.h"

@implementation Dog

- (instancetype)init
{
    if(self = [super init])
    {
        NSLog(@"小狗已被派出，%lu",self.retainCount);
    }
    return self;
}

- (void)dealloc
{
    NSLog(@"小狗回到宠物中心");
    [super dealloc];
}

@end
```
在main方法中创建dog对象，给dog发送消息

```ObjectiveC
#import <Foundation/Foundation.h>
#import "Dog.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        //模拟：宠物中心派出小狗
        Dog * dog = [[Dog alloc]init];
        
        //模拟：xiaoming需要和小狗玩耍，需要将其引用计数加1
        [dog retain];
        NSLog(@"小狗的引用计数为 %ld",dog.retainCount);
        
        //模拟：xiaoming不和小狗玩耍了，需要将其引用计数减1
        [dog release];
        NSLog(@"小狗的引用计数为 %ld",dog.retainCount);
        
        //没人需要和小狗玩耍了，将其引用计数减1
        [dog release];
        
        //将指针置nil，否则变为野指针
        dog = nil;
    }
    return 0;
}
```
输出结果为：

![结果](https://img-blog.csdnimg.cn/img_convert/b25cf4b2f5204a84591887f6dd0aef0f.png)

可以看到，引用计数帮助宠物中心很好的标记了小狗的使用状态，在完成任务的时候及时收回到宠物中心。

### 注意事项：
1. NSString 引用计数问题

```ObjectiveC
NSString* str = @"hello world!";
NSLog(@"%ld", str.retainCount);
```
结果输出为-1，这可以理解为str实际上是一个字符串常量，它存储在常量存储区，是没有引用计数的。

2. 赋值不会拥有某个对象

```ObjectiveC
Dog* dog1 = dog;
```
<font color=red>这里仅仅是指针赋值操作，并不会增加dog的引用计数，需要持有对象必须要发送retain消息</font>

3. dealloc
由于释放对象是会调用dealloc方法，因此重写dealloc方法来查看对象释放的情况，如果没有调用则会造成内存泄露。在上面的例子中我们通过重写dealloc让小狗被释放的时候打印日志来告诉我们已经完成释放。

4. 在上面的例子中，如果再增加这样一个操作:

```ObjectiveC
//没人需要和小狗玩耍了，将其引用计数减1
[dog release];
NSLog(@"%ld",dog.retainCount);
```
会发现引用计数为1，为什么不是0呢？这是因为对引用计数为1的对象release时，系统知道该对象将被回收，就不会再对该对象的引用计数进行减1操作，这样可以增加对象回收的效率。

另外，对已释放的对象发送消息是不可取的，因为对象的内存已被回收，如果发送消息时，该内存已经被其他对象使用了，得到的结果是无法确定的，甚至会造成崩溃。

## 自动释放池
当不再使用一个对象时应该将其释放，但是在某些情况下，我们很难理清一个对象什么时候不再使用（比如xiaoming和小狗玩耍结束的时间不确定），这应该如何处理？

ObjC提供autorelease方法来解决这个问题，当给一个对象发送autorelease消息时，方法会在未来某个时间给这个对象发送release消息将其释放，在这个时间段内，对象还是可以使用的。

### autorelease 的工作原理
当对象接收到autorelease消息时，它会被添加到了当前的自动释放池中，当自动释放池被销毁时，会给池里所有的对象发送release消息。这里就引出了自动释放池这个概念，顾名思义，就是一个池，这个池可以容纳对象，而且可以自动释放，这就大大增加了我们处理对象的灵活性。

ObjC提供两种方法创建自动释放池：

1. 使用NSAutoreleasePool来创建

```ObjectiveC
NSAutoreleasePool * pool = [[NSAutoreleasePool alloc]init];
//这里写代码
[pool release];
```
2. 使用@autoreleasepool创建

```ObjectiveC
@autoreleasepool {
//这里写代码
}
```
自动释放池创建后，就会成为活动的池子，释放池子后，池子将释放其所包含的所有对象。以上两种方法推荐第二种，因为第二种方法效率更高。

#### 自动释放池什么时候创建?
自动释放池的销毁时间是确定的，一般是在程序事件处理之后释放，或者由我们自己手动释放。
下面举例说明自动释放池的工作流程：
场景：现在xiaoming和xiaohong都想和小狗一起玩耍，但是他们的需求不一样，他们的玩耍时间不一样。

```ObjectiveC
//创建一个自动释放池
NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
//模拟：宠物中心派出小狗
Dog * dog = [[Dog alloc]init];

//模拟：xiaoming需要和小狗玩耍，需要将其引用计数加1
[dog retain];
NSLog(@"小狗的引用计数为 %ld",dog.retainCount);

//模拟：xiaohong需要和小狗玩耍，需要将其引用计数加1
[dog retain];
NSLog(@"小狗的引用计数为 %ld",dog.retainCount);

//模拟：xiaoming确定不想和小狗玩耍了，需要将其引用计数减1
[dog release];
NSLog(@"小狗的引用计数为 %ld",dog.retainCount);

//模拟：xiaohong不确定何时不想和小狗玩耍了，将其设置为自动释放
[dog autorelease];
NSLog(@"小狗的引用计数为 %ld",dog.retainCount);

//没人需要和小狗玩耍了，将其引用计数减1
[dog release];

NSLog(@"释放池子");
[pool release];
```
结果如下：

![结果](https://img-blog.csdnimg.cn/img_convert/08148f61a2a6260ab599542d6a87b8fe.png)

可以看到，当池子释放后，dog对象才被释放，因此在池子释放之前，xiaohong都可以尽情地和小狗玩耍。

#### 使用自动释放池需要注意
1. 自动释放池实质上只是在释放的时候給池中所有对象对象发送release消息，不保证对象一定会销毁，如果自动释放池向对象发送release消息后对象的引用计数仍大于1，对象就无法销毁。
2. 自动释放池中的对象会集中同一时间释放，如果操作需要生成的对象较多占用内存空间大，可以使用多个释放池来进行优化。比如在一个循环中需要创建大量的临时变量，可以创建内部的池子来降低内存占用峰值。
3. <font color=red>autorelease不会改变对象的引用计数</font>

#### 自动释放池的常见问题
在管理对象释放的问题上，自动帮助我们释放池节省了大量的时间，但是有时候它却未必会达到我们期望的效果，比如在一个循环事件中，如果循环次数较大或者事件处理占用内存较大，就会导致内存占用不断增长，可能会导致不希望看到的后果。

示例代码:
```ObjectiveC
for (int i = 0; i < 100000; i ++) {
    NSString * log  = [NSString stringWithFormat:@"%d", i];
    NSLog(@"%@", log);
}
```
前面讲过，自动释放池的释放时间是确定的，这个例子中自动释放池会在循环事件结束时释放，那问题来了：在这个十万次的循环中，每次都会生成一个字符串并打印，这些字符串对象都放在池子中并直到循环结束才会释放，因此在循环期间内存不增长。

这类问题的解决方案是在循环中创建新的自动释放池，多少个循环释放一次由我们自行决定。

```ObjectiveC
for (int i = 0; i < 100000; i ++) {
    @autoreleasepool {
        NSString * log  = [NSString stringWithFormat:@"%d", i];
        NSLog(@"%@", log);
    }
}
```

## iOS的内存管理规则

### 基本规则
1. 当你通过new、alloc或copy方法创建一个对象时，它的引用计数为1，当不再使用该对象时，应该向对象发送release或者autorelease消息释放对象
2. 当你通过其他方法获得一个对象时，如果对象引用计数为1且被设置为autorelease，则不需要执行任何释放对象的操作
3. 如果你打算取得对象所有权，就需要保留对象并在操作完成之后释放，且必须保证retain和release的次数对等

应用到文章开头的例子中，小朋友每申请一个小狗（生成对象），最后都要归还到宠物中心（释放对象），如果只申请而不归还（对象创建了没有释放），那宠物中心的小狗就会越来越少（可用内存越来越少），到最后一个小狗都没有了（内存被耗尽），其他小朋友就再也没有小狗可申请了（无资源可申请使用），因此，必须要遵守规则：申请必须归还（规则1），申请几个必须归还几个（规则3），如果小狗被设定归还时间则不用小朋友主动归还（规则2）

### ARC
在MRC时代，必须严格遵守以上规则，否则内存问题将成为恶魔一样的存在，然而来到ARC时代，事情似乎变得轻松了，不用再写无止尽的ratain和release似乎让开发变得轻松了，对初学者变得更友好。

ObjC2.0引入了垃圾回收机制，然而由于垃圾回收机制会对移动设备产生某些不好的影响（例如由于垃圾清理造成的卡顿），iOS并不支持这个机制，苹果的解决方案就是ARC（自动引用计数）。

iOS5以后，我们可以开启ARC模式，ARC可以理解成一位管家，这个管家会帮我们向对象发送retain和release语句，不再需要我们手动添加了，我们可以更舒心地创建或引用对象，简化内存管理步骤，节省大量的开发时间。

实际上，ARC不是垃圾回收，也并不是不需要内存管理了，它是隐式的内存管理，编译器在编译的时候会在代码插入合适的ratain和release语句，相当于在背后帮我们完成了内存管理的工作。

下面将自动释放池的例子转化为ARC来看看：
```ObjectiveC
@autoreleasepool {

    Dog * dog = [[Dog alloc]init];

    [xiaoming playWithDog:dog];

    [xiaohong playWithDog:dog];

    NSLog(@"释放池子");
}
```
#### 注意事项
- 如果你的工程历史比较久，可以将其从MRC转换成ARC，跟上时代的步伐更好地维护
- 如果你的工程引用了某些不支持ARC的库，可以在Build Phases的Compile Sources将对应的m文件的编译器参数配置为-fno-objc-arc
- <font color=red>ARC能帮我们简化内存管理问题，但不代表它是万能的，还是有它不能处理的情况，这就需要我们自己手动处理，比如循环引用、非ObjC对象、Core Foundation中的malloc()或者free()等等</font>

### ARC的修饰符
ARC提供四种修饰符，分别是：
- `strong`, 
- `weak`, 
- `autoreleasing`, 
- `unsafe_unretained`

下面举例说明：
- `_strong` : 强引用，持有所指向对象的所有权，无修饰符情况下的默认值。如需强制释放，可置nil。

比如我们常用的定时器：
```ObjectiveC
NSTimer * timer = [NSTimer timerWith...];
```
相当于
```ObjectiveC
NSTimer * __strong timer = [NSTimer timerWith...];
```
当不需要使用时，强制销毁定时器
```ObjectiveC
[timer invalidate];
timer = nil;
```
- `_weak` : 弱引用，不持有所指向对象的所有权，引用指向的对象内存被回收之后，引用本身会置nil，避免野指针。

比如避免循环引用的弱引用声明：
```ObjectiveC
__weak __typeof(self) weakSelf = self;
```
- `__autoreleasing` : 自动释放对象的引用，一般用于传递参数

比如一个读取数据的方法:
```ObjectiveC
- (void)loadData:(NSError **)error;
```
当你调用时会发现这样的提示
```ObjectiveC
NSError * error;
[dataTool loadData:(NSError *__autoreleasing *)]
```
这时编译器自动帮我们插入以下代码：
```ObjectiveC
NSError * error;
NSError * __autoreleasing tmpErr = error;
[dataTool loadData:&tmpErr];
```
- `__unsafe_unretained` : 为兼容iOS5以下版本的产物，可以理解成MRC下的weak，现在基本用不到，这里不作描述。

### 属性的内存管理
ObjC2.0引入了@property，提供成员变量访问方法、权限、环境、内存管理类型的声明，下面主要说明ARC中属性的内存管理。

属性的参数分为三类，基本数据类型默认为(atomic,readwrite,assign)，对象类型默认为(atomic,readwrite,strong)，其中第三个参数就是该属性的内存管理方式修饰，修饰词可以是以下之一：

- `assign` : 一般用来修饰基本数据类型
```ObjectiveC
@property (nonatomic, assign) NSInteger count;
```
当然也可以修饰ObjC对象，但是不推荐，因为被assign修饰的对象释放后，指针还是指向释放前的内存，在后续操作中可能会导致内存问题引发崩溃。

- `retain` : release旧值，再retain新值（引用计数＋1）

retain和strong一样，都用来修饰ObjC对象。使用set方法赋值时，实质上是会先保留新值，再释放旧值，再设置新值，避免新旧值一样时导致对象被释放的的问题。

MRC写法如下：
```ObjectiveC
- (void)setCount:(NSObject *)count {
    [count retain];
    [_count release];
    _count = count;
}
```
ARC写法如下：
```ObjectiveC
- (void)setCount:(NSObject *)count {
    _count = count;
}
```
- `copy` : release旧值，再copy新值（拷贝内容）

一般用来修饰String、Dict、Array等需要保护其封装性的对象，尤其是在其内容可变的情况下，因此会拷贝（深拷贝）一份内容給属性使用，避免可能造成的对源内容进行改动。

使用set方法赋值时，实质上是会先拷贝新值，再释放旧值，再设置新值。
```ObjectiveC
@property (nonatomic, copy) NSString* name;
```

- `weak` : ARC新引入修饰词，可代替assign，比assign多增加一个特性（置nil，见上文）。

weak和strong一样用来修饰ObjC对象。使用set方法赋值时，实质上不保留新值，也不释放旧值，只设置新值。

比如常用的代理的声明：
```ObjectiveC
@property (weak) id<MyDelegate> delegate;
```

Xib控件的引用：
@property (weak, nonatomic) IBOutlet UIImageView* productImage;

- `strong` : ARC新引入修饰词，可代替retain

可参照retain，这里不再作描述。

### block的内存管理
iOS中使用block必须自己管理内存，错误的内存管理将导致循环引用等内存泄漏问题，这里主要说明在ARC下block声明和使用的时候需要注意的两点：

- 如果你使用@property去声明一个block的时候，一般使用copy来进行修饰（当然也可以不写，编译器自动进行copy操作），尽量不要使用retain。

```ObjectiveC
@property (nonatomic, copy) void(^block)(NSData* data);
```
- block会对内部使用的对象进行强引用，因此在使用的时候应该确定不会引起循环引用，当然保险的做法就是添加弱引用标记。
```ObjectiveC
__weak typeof(self) weakSelf = self;
```

## 经典内存泄漏及其解决方案
虽然ARC好处多多，然而也并无法避免内存泄漏问题，下面介绍在ARC中常见的内存泄漏。

### 僵尸对象和野指针
<font color=orange>僵尸对象：内存已经被回收的对象</font>

<font color=orange>野指针：指向僵尸对象的指针，向野指针发送消息会导致崩溃</font>

野指针错误形式在Xcode中通常表现为：Thread 1：EXC_BAD_ACCESS，因为你访问了一块已经不属于你的内存。

例子代码：(没有出现错误的多运行几遍，因为获取野指针指向的结果是不确定的)
```ObjectiveC
Dog * dog = [[Dog alloc]init];
NSLog(@"before");
NSLog(@"%s",object_getClassName(dog));
[dog release];
NSLog(@"after");
NSLog(@"%s",object_getClassName(dog));
```
运行结果：

![结果](https://img-blog.csdnimg.cn/img_convert/2b5c91ee80037ab5ecc3b6309ed5e131.png)

可以看到，当运行到第六行的时候崩溃了，并给出了EXC_BAD_ACCESS的提示。

#### 解决方案
对象已经被释放后，应将其指针置为空指针（没有指向任何对象的指针，给空指针发送消息不会报错）。

然而在实际开发中实际遇到EXC_BAD_ACCESS错误时，往往很难定位到错误点，幸好Xcode提供方便的工具給我们来定位及分析错误。

- 在product－scheme－edit scheme－diagnostics中将enable zombie objects勾选上，下次再出现这样的错误就可以准确定位了。
- 在Xcode－open developer tool－Instruments打开工具集，选择Zombies工具可以对已安装的应用进行僵尸对象检测。

### 循环引用
循环引用是ARC中最常出现的问题，对于可能引发循环引用的一些原因在有一篇文章[iOS总结篇：影响控制器正常释放的常见问题](http://www.jianshu.com/p/bcc0bcaadd6c)中有提及。

一般来讲循环引用也是可以使用工具来检测到的，分为两种：
- 在 product－Analyze 中使用静态分析来检测代码中可能存在循环引用的问题。
- 在 Xcode－open developer tool－Instruments 打开工具集，选择Leaks工具可以对已安装的应用进行内存泄漏检测，此工具能检测静态分析不会提示，但是到运行时才会出现的内存泄漏问题。

Leaks工具虽然强大，但是它不能检测到block循环引用导致的内存泄漏，这种情况一般需要自行排查问题（考验你的基本功时候到了）。

### 循环中对象占用内存大
这个问题常见于循环次数较大，循环体生成的对象占用内存较大的情景。

例子代码：我需要10000个演员来打仗
```ObjectiveC
for (int i = 0; i < 10000; i ++) {
    Person * soldier = [[Person alloc]init];
    [soldier fight];
}
```
该循环内产生大量的临时对象，直至循环结束才释放，可能导致内存泄漏，解决方法和上文中提到的自动释放池常见问题类似：在循环中创建自己的autoReleasePool，及时释放占用内存大的临时变量，减少内存占用峰值。
```ObjectiveC
for (int i = 0; i < 10000; i ++) {
    @autoreleasepool {
        Person * soldier = [[Person alloc]init];
        [soldier fight];
    }
}
```

然而有时候autoReleasePool也不是万能的：

例子：假如有2000张图片，每张1M左右，现在需要获取所有图片的尺寸，你会怎么做？
```ObjectiveC
for (int i = 0; i < 2000; i ++) {
    CGSize size = [UIImage imageNamed:[NSString stringWithFormat:@"%d.jpg",i]].size;
    //add size to array
}
```
用imageNamed方法加载图片占用Cache的内存，autoReleasePool也不能释放，对此问题需要另外的解决方法，当然保险的当然是双管齐下了:
```ObjectiveC
for (int i = 0; i < 2000; i ++) {
    @autoreleasepool {
        CGSize size = [UIImage imageWithContentsOfFile:filePath].size;
        //add siez to array
    }
}
```

### 无限循环
无论你出于什么原因，当你启动了一个无限循环的时候，ARC会默认该方法用不会执行完毕，方法里面的对象就永不释放，内存无限上涨，导致内存泄漏。

例子：
```ObjectiveC
NSLog(@"start !");
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    BOOL isSucc = YES;
    while (isSucc) {
        [NSThread sleepForTimeInterval:1.0];
        NSLog(@"create an obj");
    }
});
```
对于这样的问题还是很好解决的，只要设置相关条件判断就能避免。

## 总结
关于iOS内存管理的知识点还有很多，在这里先做基础储备，后续在实战中会继续补充。

___
## 参考
- [iOS内功篇：内存管理](http://www.jianshu.com/p/8b1ed04b3ba9)
