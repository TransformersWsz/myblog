---
title: ThinkPHP多表回滚无效
mathjax: true
toc: true
date: 2018-04-15 16:57:34
categories:
- Software Engineering
tags:
- PHP
- ThinkPHP
- 多表回滚
---
今天首次用到了多表回滚，遇到了一个坑，记录一下。

<!--more-->

<font color="red">错误代码如下：</font>
```php
try{
    $Member = D("Member");
    $Member->startTrans();
    $member_condition['id'] = 11641;
    $member_data['id'] = 10000;
    $member_res = $Member->where($member_condition)->save($member_data);

    if ($member_res === 1) {
        try{
            $User = D("User");
            $User->startTrans();
            $user_condition['account'] = '111111';
            $user_data['username'] = "4324";
            $user_res = $User->where($user_condition)->save($user_data);
            if ($user_res === 1) {
                $User->commit();
                $Member->commit();
                echo "全部修改成功";
            }
            else {
                $User->rollback();
                $Member->rollback();
                echo "User表未受影响，全部回滚！";
            }
        }catch (Exception $e){
            $User->rollback();
            $Member->rollback();
            echo "User修改异常，全部回滚！";
        }
    }
    else {
        $Member->rollback();
        echo "Member表未受影响，回滚！";
    }
}catch (Exception $e){
    $Member->rollback();
    echo "Member修改异常！";
}
```
我的思路是对 `Member` 和 `User` 分别开启事务，只要有一个表修改失败，那么就全部回滚。但事实确是开启了两个事务后，这两个事务都无法回滚。如果只开启一个事务，那么该事务是可以回滚的。在tp官方文档里面也没找到什么解释。解决方法如下所示：
```php
try{
    $Model = M();
	$Model->startTrans();
	$member_condition['id'] = 11641;
    $member_data['id'] = 10000;
    $member_res = $Model->table('party_member')->where($member_condition)->save($member_data);

    $user_condition['account'] = '111111';
    $user_data['username'] = "4324";
    $user_res = $Model->table('party_user')->where($user_condition)->save($user_data);

    if ($member_res === 1 && $user_res === 1) {
	    echo "commit";
        $Model->commit();
    }
    else{
        echo "rollback";
        $Model->rollback();
    }
}catch (Exception $e){
    echo "发生异常";
    $Model->rollback();
}
```
对于多表的事务处理，先用 M 函数实例化一个空对象，使用 table 方法进行多个表的操作，如果操作成功则提交，失败则回滚。

另外一点需要说明的是，在有些集成环境中MySQL默认的引擎是 `MyISAM`，若想提供事务支持，需将数据库引擎改为 `InnoDB` 。