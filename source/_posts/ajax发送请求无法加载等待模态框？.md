---
title: ajax发送请求无法加载等待模态框？
mathjax: true
toc: true
date: 2018-07-15 23:54:57
categories:
- Software Engineering
tags:
- Ajax
---
虽说现在谈论jQuery已经很low了，但出于维护旧项目的需要，还是重新学习了一遍。当我们向后台发送请求的时候，为了照顾用户体验，需要使用等待模态框框来过渡。今天遇到的一个坑是，无论怎么发送请求，界面都不会出现模态框，即使有也是一闪而过。代码如下：

<!--more-->

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>test</title>
    <script src="./js/jquery-3.1.1.js" type="text/javascript"></script>
    <link href="./css/bootstrap.min.css" type="text/css" rel="stylesheet">
    <script src="./js/bootstrap.min.js" type="text/javascript"></script>
    <script>

        function firmSubmit() {
            $("#submit").click(function () {
                var sex = parseInt($('#sex').val());
                if (sex == 0) {
                    alert("性别不能为空！");
                    $('#sex').focus();
                    return false;
                }

                $.ajax({
                    type: "post",
                    url: "/api",
                    dataType: "json",
                    data: { "sex": sex },
                    beforeSend: function () {
                        $("#loadingModal").modal("show");
                    },

                    success: function (json) {
                        if (json.result == 1) {
                            alert("提交成功");
                        }
                        else {
                            alert("提交失败，请重新检查！");
                        }
                    },
                    complete: function () {
                        $("#loadingModal").modal("hide");
                        window.location.href = "/in";
                    }
                });
            });
        }
        $(function () {
            //使用getJSON方法读取json数据,
            var options = ["sex"];
            $.ajaxSetup({
                async: false
            });
            $.getJSON("/getsex", function (data) {
                for (var i = 0; i < options.length; i++) {
                    $.each(data[options[i]], function (key, val) {
                        var str = "<option value=" + key + ">" + val + "</option>";
                        $("#" + options[i]).append(str);
                    });
                }
            });

            //表单判空并提交
            firmSubmit();

        });
    </script>
    <style>
        .main {
            width: 38%;
        }

        .commonlabel {
            font-size: 16px;
            margin-left: 5px;
        }

        .loading {
            height: 80px;
            width: 80px;
            background: url('./img/load.gif') no-repeat center;
            opacity: 0.7;
            position: fixed;
            left: 50%;
            top: 50%;
            margin-left: -40px;
            margin-top: -40px;
            z-index: 1001;
            background-color: #dad8d8;
            -moz-border-radius: 20px;
            -webkit-border-radius: 20px;
            border-radius: 20px;
            filter: progid:DXImageTransform.Microsoft.Alpha(opacity=70);
        }
    </style>

</head>

<body>
    <div id="loadingModal" class="modal fade" data-keyboard="false" tabindex="-1" data-backdrop="static" data-role="dialog" aria-labelledby="myModalLabel"
        aria-hidden="true">
        <div id="loading" class="loading"></div>
    </div>

    <div class="main center-block">
        <p style="height: 30px;"></p>

        <div class="form-group">
            <label for="sex" class="commonlabel">性别</label>
            <select id="sex" class="form-control" name="sex"></select>
        </div>

        <button id="submit" class="btn btn-primary" style="width: 20%; margin: 0 auto; display: block; float: left;">提交</button>
    </div>

</body>

</html>
```
后来才发现是 `$.ajaxSetup({async: false});` 搞的鬼。查了一下官方文档，它是这样解释的：
> async (default: true)
Type: Boolean
By default, all requests are sent asynchronously (i.e. this is set to true by default). If you need synchronous requests, set this option to false. Cross-domain requests and dataType: "jsonp" requests do not support synchronous operation. Note that synchronous requests may temporarily lock the browser, disabling any actions while the request is active. As of jQuery 1.8, the use of async: false with jqXHR ($.Deferred) is deprecated; you must use the success/error/complete callback options instead of the corresponding methods of the jqXHR object such as jqXHR.done().

也就是说设置 `async: false` 会锁住浏览器，禁止浏览器的任何行为。比如用户点击按钮、下拉滚动条等行为浏览器都不会有响应，直到该同步请求完成。所以发送同步请求期间，模态框不会出现。但在请求完成后，会执行：
```html
complete: function () {
    $("#loadingModal").modal("hide");
    window.location.href = "/in";
}
```
因此会出现模态框一闪而过的情况。解决方法也很简单，在发送请求前将 `$.ajaxSetup({async: true});` 设置回来即可。