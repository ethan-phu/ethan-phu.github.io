# 关于周报这种小事


从认识到双向链接开始我先使用了 obsidian,然而对于我这种懒癌晚期的人来说，需要结构化的记录，真的不太适合我「取一个好听的名字真的太难了」。当然有一说一，双链这个观点真的是太妙了。

现在的我已经全面转向了 logseq 来进行笔记记录，不得不说这种支持自定义代码的笔记真的不错「虽然我到目前为止使用最多的是**高级查询**和**TODO**这两个功能，记录笔记的话只是零星记录了几句话，并没有详细的记录或者输出一些东西。」。在 logseq 中是以日期为主线的，这免去了我对要写的内容的抽象主题的负担。

## 工作日报周报

首先对任务进行分类，在学习了 Logseq 的高级查询语法并了解其能力之后，我决定使用`#tag`这种形式来组织个人任务和工作任务，属于工作任务的会标记上`#work`标签，个人任务就不进行标签。

首先就是工作，这个是大块内容，我目前使用 logseq 生成**日报&周报**，用于之后提交到绩效评估系统中。

### 日报/周报代码

日报和周报代码上只有 inputs[:yesterday]更改成 inputs[:7d]即可。

```logseq
#+BEGIN_QUERY
{
    :title "查询昨天完成的任务"
    :query [:find (pull ?b [*])
       :in $ ?start ?end
       :where
       [?b :block/marker ?m]
       [?b :block/page ?p]
       [?p :page/journal? true]
       [?p :page/journal-day ?d]
       [(>= ?d ?start)]
       [(<= ?d ?end)]
       [?b :block/path-refs [:block/name "work"]]
       [(= "DONE" ?m)]
   ]
   :inputs [:yesterday :today]
}
#+END_QUERY
```

效果图如下：
![logseq周报效果图](https://ethanphu.oss-cn-beijing.aliyuncs.com/winebrennerian-preventively-mellate-parisianly.png)

