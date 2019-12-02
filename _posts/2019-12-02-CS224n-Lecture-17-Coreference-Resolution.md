---
layout:     post
title:      CS224n Lecture 17 Coreference Resolution
subtitle:   Lecture Note
date:       2019-12-02
author:     viewsetting
header-img: img/post-adjani.jpg
catalog: true
tags:
   - NLP
   - CS224n
   - Notes
---

# 指代消解的步骤

指代消解任务可以分为两步式的任务：

- 检测Mention的边界
- 对于Mention进行聚类

对于第一个任务，举个例子：

> "I voted for Nader because he was most aligned with my values," she said

在这一句中，如果只是做mention检测的任务的话我们得到的结果应该是：

> "[I] voted for [Nader] because [he] was most aligned with [ [my] values]," [she] said

在这里所有的mention都被识别了，其中my values这个实体还存在嵌套的现象，即my指代的是之前的I。

但是一个完整的指代消解系统接下开还需要做的就是分类这些Mention，如果将之视为一个聚类问题的话，显然地，可以将指代相同实体的Mention聚类到一起。

对于指代消解任务而言，对于Mention的聚类比实体识别（即边界检测）任务要难多了。

# 边界检测

对于边界检测而言，我们又可以从Mention的种类出发进行任务的分解：

|    种类     |      Pronoun       |          Named Entity           |      Noun Phase      |
| :---------: | :----------------: | :-----------------------------: | :------------------: |
|    举例     | I, she, he, you... | People, Places, Orgnazations... |       "a dog"        |
| 对应NLP任务 |     POS tagger     |               NER               | Constitutency Parser |

在进行这些任务的时候，我们又会遇到一些特例：

*It is funny.* 

*Every Student*

*No Student*

*500 miles*

上面的这些特例里，有it这种特殊的指代方式，如代指心情，天气等等。还有every, no等绝对意义的名词短语，很显然every student不是一个具体的实体的一种指代。当然对于*500 miles*这种表示数量的名词短语也不会是指代。

那么我们就有两种策略：

- 第一种就是先构造一个classifier，将所有的mention candidates筛查一遍，去除其中的特例。
- 第二种自然就是将所有可能的mention全都保留。

然后如果我们考虑第二种策略的话，从更加宏观的角度来看，我们又应该如何选择mention detection的策略呢。简单的想自然就是将上述三种NLP任务对应的模型组合起来就好了，但是在后面我们还能做到真正的端到端coref，即不单独对于某个部分设计模型。

# 语言学知识的补充

在coref resolution中，我们经常会考虑两种语言现象，共指和指代。

对于指代而言，我们更常见的是anaphora。

## anaphora & coreference

举个例子：

对于以下的句子：

> Barack Obama said he would sign the bill, Barack was ...

那么coref和anaphora的对比图如下：

![](/Users/viewsetting/Pictures/Photos Library.photoslibrary/resources/renders/3/37FEC134-6DCF-4577-BE4F-9FC13D4367D7_1_201_a.jpeg)

上图展示了coref和anaphora的区别，其中OBAMA表示的是现实世界的前美国总统巴拉克奥巴马，在coref中，就是后半句的Obama和句子开始位置的Barack Obama同时指向了现实世界的实体奥巴马，而anaphora则是前半句中的he指代的就是前面出现的Barack Obama这一Mention，而Barack Obama则会产生一个实体link到实体奥巴马中。

## cataphora

cataphora可以看作为anaphora的逆操作，因为一般情况下，我们都是使用代词来指代前面的某个具体的mention，然而反方向的指代在语言中也是存在的。

如： 奥斯卡王尔德的《道雷格林的画像》选段

> “$From$ the corner of the divan of Persian saddle-bags on which **he** was lying, smoking, as was **his** custom, innumerable cigarettes, **Lord Henry Wotton** could just catch the gleam of the honey-sweet and honey-coloured blossoms of a laburnum, whose tremulous branches seemed hardly able to bear the burden of a beauty so flamelike as theirs; and now and then the fantastic shadows of birds in flight flitted across the long tussore-silk curtains that were stretched in front of the huge window, producing a kind of momentary Japanese effect, and making him think of those pallid, jade-faced painters of Tokyo who, through the medium of an art that is necessarily immobile, seek to convey the sense of swiftness and motion. “

![Oscar Wilde](https://images.gr-assets.com/authors/1521044377p4/3565.jpg)

在上面的选段里，“Lord Henry Wotton"才是前句中的he以及his的指代。

## anaphora的特例

对于我们要做的有意义的coref resolution而言，有些anaphora其实也不应该被算入最后的结果。

> No dancer twisted her knee.

对于her而言，很显然这个词指代的是“no dancer”，然而对于现实任务而言这显然是毫无意义的，因为no dancer并不能够对应现实世界的一个实体。

如果将coref和anaphora两个集合画出文恩图的话，coref和anaphora的交集是pronominal anaphora，而anaphora的剩下的集合就是bridging anaphora。

举个例子：

> We went to see a movie yesterday.
>
> The tickets were expensive.

在上面的两句话中，  the tickets指代的就是上一句的a movie，但是这两个mention之间却并没有一个coref的关系。

# 模型种类

总结下来有四种实现coreference resolution的方案：

- rule-based 如：Hobbs naive algorithm
- 基于Mention Pair
- 基于Mention Ranking
- 聚类

# 评价指标

$B^3$ eval metric

将结果看为一种聚类的话，对于预测出来的一种聚类集合，