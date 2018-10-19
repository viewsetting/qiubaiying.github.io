---
layout:     post
title:     TensorBoard在win10下的使用实例
subtitle:   Linux废物说：真香！
date:       2018-10-19
author:     viewsetting
header-img: img/post-adjani.jpg
catalog: true
tags:
    - TensorFlow
    - Green Hand Tutorial
    - Trick
    
---

## 示例代码：

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
writer = tf.summary.FileWriter(r'E:\Python\Test',tf.get_default_graph())
writer.close()
```

![1539912067565](/img/2018-10-19/pycharm.png)



## 运行TensorBoard

![1539912169718](/img/2018-10-19/cmd.png)



## 打开浏览器

![1539912245848](/img/2018-10-19/tfb.png)



#### 复制了地址：计算机名（即localhost) +6006端口 ，就能愉快玩耍了。

