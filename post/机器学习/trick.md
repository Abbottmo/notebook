# code

```python
字符串分割

line = line.strip().replace('\n', '').replace('\r', '')  ##去除首位空格 ,
##linux 换行\n  windows 换行 \n\r  去除

data = line.split(" ")##单空格分割

data = line.split() ##d多空格分割

numpy

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

           

```

### torch

```python

```

