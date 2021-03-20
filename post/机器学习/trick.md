# code

```python
字符串分割

line = line.strip().replace('\n', '').replace('\r', '')  ##去除首位空格 ,
##linux 换行\n  windows 换行 \n\r  去除

data = line.split(" ")##单空格分割

data = line.split() ##d多空格分割


def letter_box_image(img, w, h, value):
    # 将图像resize到w*h，保持原图像的长宽比，填充像素的像素值为value
    img_h, img_w = img.shape[0], img.shape[1]
    dim = max(img_h, img_w)
    ratio = w / dim
    resize_h = int((img_h / dim) * w)
    resize_w = int((img_w / dim) * h)
    if ((w - resize_w) % 2):
        resize_w -= 1
    if ((h - resize_h) % 2):
        resize_h -= 1
    offset_w = int((w - resize_w) / 2)
    offset_h = int((h - resize_h) / 2)
    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
    resize_img = cv2.copyMakeBorder(img, offset_h, offset_h, offset_w, offset_w, cv2.BORDER_CONSTANT, value=value)
    # return resize_img, ratio, offset_w, offset_h
    return resize_img

## 生成分割mask代码
mask = np.zeros((height, width), dtype=np.uint8)
mask = PIL.Image.fromarray(mask)
draw = PIL.ImageDraw.Draw(mask)
#shapes 保存分割的边界点，使用polgon围起来填充
for i in range(len(shapes)):
    shape = shapes[i]
    shape_type = shape['shape_type']
    label = shape['label']
    if shape_type == 'polygon' and ("red_pin_head" in label):
        points = shape['points']
        xy = [tuple(point) for point in points]

        draw.polygon(xy=xy,fill=1)

        if shape_type == 'polygon' and "black_pin_head" in label:
            points = shape['points']
            xy = [tuple(point) for point in points]
            ##draw.polygon(xy=xy, outline=1, fill=255) ## 不可以使用 outline ，会导致边界分割错误
            draw.polygon(xy=xy, fill=2)
```

### numpy

```python
grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0), axis=0)
grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0), axis=0)


numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
在指定的间隔内返回均匀间隔的数字。
返回num均匀分布的样本，在[start, stop]。
这个区间的端点可以任意的被排除在外。
np.linspace(1, 10, 10)     array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])



numpy.repeat
(1)numpy.repeat(a,repeats,axis=None); (2)object(ndarray).repeat(repeats,axis=None):理解了第一种方法就可以轻松知道第二种方法了。
参数的意义：
axis=None，时候就会flatten当前矩阵，实际上就是变成了一个行向量
axis=0,沿着y轴复制，实际上增加了行数
axis=1,沿着x轴复制，实际上增加列数
repeats可以为一个数，也可以为一个矩阵

x = np.array([[1,2],[3,4]])
np.repeat(x, 2)  
               得到 array([1, 1, 2, 2, 3, 3, 4, 4]) #每个元素重复两次  变成flatten矩阵
np.repeat(x, 3, axis=1)    
				得到array([[1, 1, 1, 2, 2, 2],
                         [3, 3, 3, 4, 4, 4]])   #每个元素按照列重复3次
np.repeat(x, [1, 2], axis=0)  
				array([[1, 2],
               		  [3, 4],
       			      [3, 4]])  #第1行元素重复1次，第2行元素重复2次

           
numpy.argmax
#对于多个通道的分割预测图 output =  [n,s,s] 在s*s 图上有n个通道,共n个类别
predict = numpy.argmax(output,0) ##[s,s]，在每个位置取通道最大的位置作为分割的类别 0，1，2，3...n-1
res = []
for i in  range(0,n):
    temp = np.zeros(predict.shape)
    temp[predict == i] = 255
    res.append(temp)
res = np.array(res) ## n 张分割图

numpy  扩展维度
可以通过在切片中增加None或者np.newaxis实现，它们的作用就是在相应的位置上增加一个维度，在这个维度上只有一个元素。
>>> x.shape
(2, 3, 1)
y = x[None]
>>> y.shape
(1, 2, 3, 1)

 a = x[:,None,:,:]
a.shape
(2, 1, 3, 1)
#最后一个维度添加新的维度
img = img[...,np.newaxis]

np.expand_dims()  

```

### torch

```python
https://download.pytorch.org/whl/torch_stable.html
下载稳定版本，需要vpn

torch.clamp()  
将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
同 numpy.clip()

torch.save()
1.6版本以后save 使用  _use_new_zipfile_serialization=False  兼容历史版本的torch.load()
否则会报错


```

#### opencv

```
从x86_64 + ubuntu14.04 + python3.5中import cv2(opencv3.3), 遇到以下错误：

ImportError: libSM.so.6: cannot open shared object file: No such file or directory
ImportError: libXrender.so.1: cannot open shared object file: No such file or directory
ImportError: libXext.so.6: cannot open shared object file: No such file or directory
安装对应的软件包解决：
apt-get install libsm6
apt-get install libxrender1
apt-get install libxext-dev

```



##### python

```python

## https://zhuanlan.zhihu.com/p/124870204
    
1. Python的列表为什么慢
Python的列表是一个动态的数组，即数组的size是可以调整的，数组存放着指向各个列表元素的指针(PyObject*)。列表中的各个元素可以是不同的类型，比如my_list = ['a', 1, True]。实际上数组里存放了三个指针，分别指向了这三个元素。那么相比其他语言的数组而言，为什么Python的列表会慢呢？原因主要是以下两个：

1. Python是动态类型语言，意味着类型检查要耗费额外的时间。

2. Python或者说Cpython没有JIT优化器。

2. 如何用Python执行快速的数组计算
目前比较主流的解决方案有如下几种：

1. Numpy - Numpy的array更像是C/C++的数组，数据类型一致，而且array的方法(如sum)都是用C来实现的。

2. Numba - 使用JIT技术，优化Numpy的性能。无论是调用Numpy的方法，还是使用for循环遍历Numpy数组，都可以得到性能提升。

3. Numexpr - 避免Numpy为中间结果分配内存，优化Numpy性能，主要用于大数组的表达式计算。

4. Cython - 为Python编写C/C++扩展。

接下来通过两个例子来演示如何通过这四种工具


快速的计算方式
numpy
import numpy as np
def sqr_sum(arr):
    return (arr ** 2).sum()

arr = np.array(arr)
print("The result is:", sqr_sum(arr))
9.66 µs 

numba 
from numba import jit
@jit(nopython=True)
def sqr_sum(arr):
    return (arr ** 2).sum()

arr = np.array(arr)
print("The result is:", sqr_sum(arr))
%timeit sqr_sum(arr)
The result is: 333283335000
3.39 µs

Cython

cimport numpy as np
ctypedef np.int_t DTYPE_t

def sqr_sum(np.ndarray[DTYPE_t] arr):
    cdef:
        DTYPE_t total = 0
        DTYPE_t x
        int i = 0
        int n = len(arr)
    while i < n:
        total += arr[i] ** 2
        i += 1
    return total
arr = np.array(arr, dtype="int")
print("The result is:", sqr_sum(arr))

The result is: 333283335000
5.51 µs 
```



#### 解决windows 使用pytorch 的dataloader功能时 出现pipe error的问题  

![image-20210320002523281](C:\Users\Administrator\Desktop\notebook\post\机器学习\img\image-20210320002523281.png)

但是我的电脑还是会出现这种错误，待测试，增加num_workers 会提高训练效率

