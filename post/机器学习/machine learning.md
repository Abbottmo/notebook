# 机器学习

## 凸集 凸函数 判定凸函数

![](img/image-20201109214420815.png)

AI 模型 = 模型 + 优化

#### 凸集

![](img/image-20201109214535970.png)

![](img/image-20201109214654492.png)

两个凸集的交集也是凸集



![](img/image-20201109214928968.png)

#### convex function 凸函数

![](img/image-20201109215027302.png)

二阶导数大于0的函数

![](img/image-20201109215313263.png)

如果是个矩阵，则矩阵二阶导数是半正定矩阵 才是凸函数

A是n阶方阵，如果对任何非零向量X，都有X'A*X≥0*，其中*X‘'*表示X的转置，就称A为**半正定矩阵**。



线性函数是凸函数

![](img/image-20201109215558317.png)

二次方函数

![](img/image-20201109215659772.png)





## transportation problem

![](img/image-20201109223832479.png)

1. 变量 decision variable
2. 目标 objective
3. 限制 constant
4. 判断目标类型
5. 寻找solver

优化问题函数库 **cvxopt.org**



## Set Cover Problem

![image-20201109224449630](img/image-20201109224449630.png)

![](img/image-20201109225301050.png)

constraint 中，对于Si中的每一个元素，都必须存在>=1 

如何转化

因为Xi 是离散的，不能通过线性优化问题解决，此时的思路是对xi范围做一个relaxation，令 xi 为离散变量，最后根据求得的xi值变换到0 或者1上
$$
0 <= x_i <= 1
$$
![](img/image-20201109230312022.png)



## duality 

standard form problem (not necessarily convex)
minimize f0(x)
subject to fi(x) ≤ 0, i = 1, . . . ,m
hi(x) = 0, i = 1, . . . , p

![](img/image-20201109233124702.png)