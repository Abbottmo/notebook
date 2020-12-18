# 数学矩阵基础

## 方差与协方差

在统计学中，**方差**是用来度量**单个随机变量**的**离散程度**，而协方差则一般用来刻画**两个随机变量**的**相似程度**，其中，**方差**的计算公式为

![image-20201130164548801](img/image-20201130164548801.png)

**协方差**的计算公式被定义为

![image-20201130164626441](img/image-20201130164626441.png)

![image-20201130164849479](img/image-20201130164849479.png)





已经根据关键点定位出了landmarks关键点  （n,2）维度

如何拟合圆

```python
class Points2Circle(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_m = np.mean(x)
        self.y_m = np.mean(y)

    def calc_r(self, xc, yc):
        return np.sqrt((self.x - xc)**2 + (self.y - yc)**2)

    def fun(self, c):
        ri = self.calc_r(*c)
        return ri - ri.mean()

    def process(self):
        center_estimate = self.x_m, self.y_m
        center = optimize.leastsq(self.fun, center_estimate)##最小二乘拟合
        center = center[0]
        r = self.calc_r(*center)
        r = r.mean()
        return center, r #返回圆心中心点和半径
```

