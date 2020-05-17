---
layout:     post
title:      矩阵求导的Trace Trick
subtitle:   公式推导
date:       2020-05-17
author:     viewsetting
header-img: img/post-adjani.jpg
catalog: true
tags:
     - Matrix
     - Linear Algebra

---

# 定义

对于标量函数对于矩阵的导数，可以记做：
$$
\frac {\partial f}{\partial \mathbf{X}} = [\frac{\partial f}{\partial X_{ij}}] 
$$
首先对于标量函数对于标量的微分，可以写作：
$$
df = \frac{\partial f}{\partial x} dx
$$
然后推广到标量函数对于向量的微分，可以理解为每一个向量的分量对于标量微分的贡献之和。
$$
df = \sum_{i=1}^{n}(\frac{\partial f}{\partial \mathbf{x_i}})d \mathbf{x_i} = (\frac{\partial f}{\partial \mathbf{x}})^\top d\mathbf{x}
$$
观察上面的式子，向量微分的左边是函数对于向量的偏微分的转置，从而转化为了向量的内积形式。因此，推广到矩阵$$\mathbf{X} \in m \times n$$ :
$$
df  = \sum_{i=1}^{m}\sum_{j=1}^{n}(\frac{\partial f}{\partial \mathbf{X_{ij}}})d\mathbf{x_{ij}}\\ = (\frac{\partial f}{\mathbf{X}})^\top d\mathbf{X}=tr((\frac{\partial f}{\mathbf{X}})^\top d\mathbf{X})
$$
上式中，$$tr$$表示的是矩阵的trace，迹，含义是矩阵主对角线元素之和，之所以映入矩阵的trace是因为它能带来运算的简化，将矩阵运算转换为标量计算。和向量内积相似，上面的矩阵转置乘矩阵的形式是矩阵的内积，而矩阵的内积刚好就是矩阵的trace。证明如下：

对于大小相同的矩阵$$A$$和$$B$$，大小为m*n。所以$$A^\top B$$能得到矩阵$$C$$,因此矩阵$$C$$的trace为：
$$
C_{ii} = \sum_{j=1}^{m} A^{\top}_{ij}B_{ji}
$$

$$
tr(C) = \sum_{i=1}^{n} C_{ii} = \sum_{i=1}^{n} \sum_{j=1}^{m}A^{\top}_{ij}B_{ji}
$$

又考虑到$$A^{\top}_{ij} = A^{\top}_{ji}$$，代入得：
$$
tr(C) = \sum_{j=1}^{n} C_{ii} = \sum_{j=1}^{m} \sum_{i=1}^{n}A_{ji}B_{ji}
$$


于是在等式右边得到了矩阵内积的定义，因此矩阵$$A$$和$$B$$的内积为$$A^{\top}B$$的trace。

因此，在实际运算中，我们只需要先展开函数的微分，然后在式子右边凑出$$d\mathbf{X}$$，最后将前面的余项转换为转置即为标量函数对矩阵的求导结果。

# 规则

首先给出矩阵微分的规则：

- $$d(\mathbf{X} \pm \mathbf{Y}) = d\mathbf{X} \pm d\mathbf{Y}$$ ，加减运算可以直接提出来
- $$d({\mathbf{X1}...\mathbf{Xi}...\mathbf{Xn}}) = \sum_{i=1}^{n} \mathbf{X1}...(d\mathbf{Xi})...\mathbf{Xn}$$ 矩阵乘法的微分，注意每个分量中的矩阵顺序不能改变，这是由矩阵乘法不可交换的性质决定的。
- $$d(X^\top) = (dX)^{\top}$$ 转置可以提出来
- $$ d (tr(\mathbf{X})) = tr(d\mathbf{X})$$ trace同上
- $$d(\mathbf{X}^{-1}) = -\mathbf{X^{-1}}d\mathbf{X}\mathbf{X^{-1}}$$

证明：
$$
d(XX^{-1}) = d(I)
$$

$$
(dX)X^{-1}+Xd(X^{-1}) = 0
$$

$$
X^{-1}Xd(X^{-1}) = -X^{-1}(dX)X^{-1}
$$

因此，
$$
d(X^{-1}) = -X^{-1}(dX)X^{-1}
$$


- $$d|X| = tr(X^{\#}dX)$$ $$X^{\#}$$为$$X$$的伴随矩阵
- $$d(X \circ Y) = dX\circ Y + X\circ dY$$ $$\circ$$为矩阵的内积，即逐元素积
- $$d (\sigma(X)) = \sigma^{'}(x) \circ dX; \sigma(X) = [\sigma(X_{ij})]$$ 这里是将作用于标量的函数拆出，通过对应矩阵生成函数矩阵，然后将其与矩阵的微分$dX$乘起来。

 接下来给出trace的一些性质：

- $$ x = tr(x)$$ 标量转换
- $$tr(X^\top) = tr(X)$$ 转置trace一样
- $$tr(A+B) = tr(A)+tr(B)$$ 线性拆分
- $$tr(A^\top B) = tr(B^\top A)$$ 满足交换律
- $$tr(A^\top (B \circ C)) = tr((A \circ B)^\top C)$$，这里三个矩阵$$A,B,C$$的大小相同。

# 举例

### 1

$$f = a^\top X b$$，其中：$$a \in R^{m \times 1}, b \in R^{n \times 1}, X \in R^{m \times n}$$ 

对$$f$$求微分：
$$
df = da^\top Xb +a^\top dX b+a\top Xdb
$$
由于$$a,b$$与$$X$$无关，所以一三项消去，得到：
$$
df = a^\top dX b
$$
由于$$df$$为标量，所以引入trace：
$$
df = tr(a^\top dX b)
$$
通过交换律:
$$
df = tr(ba^\top dX)
$$
由：
$$
df=tr((\frac{\partial f}{\mathbf{X}})^\top d\mathbf{X})
$$
得到:
$$
\frac{\partial f}{\mathbf{X}} = (ba^\top)^\top = ab^\top
$$

### 2

$$f = X^\top A X$$，其中：$$A \in R^{m \times n}, X \in R^{m \times n}$$

对$$f$$求微分：
$$
df = dX^\top AX +X^\top dAX+X^\top AdX
$$
消去$$dA$$，得到：
$$
df = dX^\top AX +X^\top AdX
$$
等式右边引入trace运算,同时通过trace的线性分解性质得到:
$$
df = tr(dX^\top AX) + tr(X^\top AdX)
$$
又由于trace中转置不变性，将第一项内的矩阵乘积转置后得到$$dX$$项，为：
$$
df = tr((AX)^\top dX) + tr(X^\top AdX)
$$
由于trace的线性，最后将两个项合并得到：
$$
df = tr((X^\top A+(AX)^\top) dX)
$$
所以：
$$
\frac{\partial f}{\mathbf{X}} =(X^\top (A+A^\top)^\top = (A+A^\top)X
$$

### 3 线性回归的最小二乘法求导

由定义得：
$$
\mathcal{L} =  \|Xw-b-y\|_2
$$
上式中，$$b$$为偏置量，可以看作向量，也可以看作标量，只需要将向量的每个分量赋同一个值。$$y$$是向量，表示数据的值，$$X \in R^{m\times n}$$是每个数据及其各个feature的矩阵，最后$$w$$是衡量每个数据点点$$n$$个feature对于最后结果$$y$$的贡献程度的向量。损失函数自然是使用最小二乘法优化，最后$$\frac{\partial \mathcal{L}}{\partial w}$$的零点即为最优的$$w$$。这个损失函数是L2范数也就是向量的内积，因此可以看作为两个向量的转置与自身的乘积：
$$
d\mathcal{L} = d[(Xw-b-y)^\top(Xw-b-y)]
$$

$$
d \mathcal{L} = d[(Xw-b-y)^\top](Xw-b-y) + (Xw-b-y)^\top d(Xw-b-y)
$$

第一项首先将转置提出来
$$
d \mathcal{L} = d(Xw-b-y)^\top(Xw-b-y) + (Xw-b-y)^\top d(Xw-b-y)
$$
接着去除微分项中不含$$w$$的部分：
$$
d \mathcal{L} = d(Xw)^\top(Xw-b-y) + (Xw-b-y)^\top d(Xw)
$$
接着引入trace：
$$
d \mathcal{L} = tr(d(Xw)^\top(Xw-b-y) )+tr( (Xw-b-y)^\top d(Xw))
$$
得到:
$$
d \mathcal{L} = tr((Xw-b-y)^\top d(Xw) )+tr( (Xw-b-y)^\top d(Xw))
$$
由于:
$$
d(Xw) = dX w+Xdw=Xdw
$$
代入得：
$$
d \mathcal{L} = tr((Xw-b-y)^\top Xdw)+tr( (Xw-b-y)^\top X dw)=tr(2(Xw-b-y)^\top X dw)
$$
所以
$$
\frac{\partial \mathcal{L}}{\partial w} = 2((Xw-b-y)^\top X)^\top = 2 X^\top(Xw-b-y)
$$
另上面的结果为0，得到：
$$
X^\top Xw = X^\top b + X^\top y
$$
化简得到：
$$
w = (X^\top X)^{-1}X^\top b + (X^\top X)^{-1}X^{\top}y
$$
如果不考虑偏置，$$w$$为：
$$
 (X^\top X)^{-1}X^{\top}y
$$
