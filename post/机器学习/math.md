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

根据 拟合直线画出直线

```python
[vx, vy, x, y] = cv2.fitLine(contours[max_idx], cv2.DIST_L12, 0, 0.01, 0.01)
## vx, vy 方向向量  x,y 为直线上一点

    # 控制交点值的范围
    if math.fabs(vy/vx)<1:
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        point1 = [cols - 1, righty]
        point2 = [0, lefty]
    else:
        upx = int((-y*vx/vy)+x)
        downx = int(((rows-y)*vx/vy)+x)
        point1 = [upx, 0]
        point2 = [downx, rows-1]
    cv2.line(img_show, (point1[0], point1[1]), (point2[0], point2[1]), (0, 255, 0), 1)
    
    
    
# 指针直线方程
def GeneralEquation(first_x,first_y,second_x,second_y):
    # 一般式 Ax+By+C=0
    A = second_y-first_y
    B = first_x-second_x
    C = second_x*first_y-first_x*second_y
    return A, B, C

## 求两个线段的交点  每个线段2个端点
def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return [px, py]

A, B, C = GeneralEquation(point1[0], point1[1], point2[0], point2[1])
for i in range(len(landmarks)-2):
    # 从圆心到刻度方向向量
    vector2 = [landmarks[i][0] - landmarks[-1][0], landmarks[i][1] - landmarks[-1][1]]
    # 1、指针直线方程在两相邻刻度之间；2、圆心到刻度方向向量与指针方向向量同向
    # if (landmarks[i][1]-k*landmarks[i][0]-b)*(landmarks[i+1][1]-k*landmarks[i+1][0]-b) <= 0 and (vector1[0]*vector2[0]+vector1[1]*vector2[1])>0:
    if (A*landmarks[i][0]+B*landmarks[i][1]+C) * (A*landmarks[i+1][0]+B*landmarks[i+1][1]+C) <= 0 and (vector1[0] * vector2[0] + vector1[1] * vector2[1]) > 0:
        # 相交点
        cross_point = findIntersection(point1[0], point1[1], point2[0], point2[1], landmarks[i][0], landmarks[i][1], landmarks[i+1][0], landmarks[i+1][1])
        cv2.line(img_show, (landmarks[i][0], landmarks[i][1]), (landmarks[i+1][0], landmarks[i+1][1]), (0, 0, 255), 1)
        cv2.circle(img_show, center=(int(cross_point[0]), int(cross_point[1])), radius=2, color=(255, 0, 0), thickness=-1)
        ratio = math.sqrt((landmarks[i][0]-cross_point[0])**2 + (landmarks[i][1]-cross_point[1])**2)/math.sqrt((landmarks[i+1][0]-landmarks[i][0])**2 + (landmarks[i+1][1]-landmarks[i][1])**2)
        value = scale[i] + round((scale[i + 1] - scale[i]) * ratio, 2)
        return value
```

