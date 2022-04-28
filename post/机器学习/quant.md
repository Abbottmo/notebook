# **量化**

![img](D:\notebook\post\机器学习\img\v2-5fb5dff51514ed3ec16640d92b4b21b5_b.jpg)

FP16的普遍精度是`~5.96e−8 (6.10e−5) … 65504`，而我们模型中的FP32权重有部分数值是`1e-10`级别。这样从FP32->FP16会导致部分精度丢失，从而模型的精度也会下降一些。

其实从FP32->FP16也是一种量化，只不过因为FP32->FP16几乎是无损的(CUDA中使用`__float2half`直接进行转换)，不需要`calibrator`去校正、更不需要`retrain`。

而且FP16的精度下降对于大部分任务影响不是很大，甚至有些任务会提升。NVIDIA对于FP16有专门的Tensor Cores可以进行矩阵运算，相比FP32来说吞吐量提升一倍。

![img](img\v2-0893272bd4a45b3a40b845928c4ed4ec_b.jpg)

**量化就是将我们训练好的模型，不论是权重、还是计算op，都转换为低精度去计算**。因为FP16的量化很简单，所以实际中我们谈论的量化更多的是**INT8的量化**，当然也有3-bit、4-bit的量化，不过目前来说比较常见比较实用的，也就是INT8量化了

那么经过INT8量化后的模型：

- 模型容量变小了，这个很好理解，FP32的权重变成INT8，大小直接缩了4倍
- 模型运行速度可以提升，实际卷积计算的op是INT8类型，在特定硬件下可以利用INT8的指令集去实现高吞吐，不论是GPU还是INTEL、ARM等平台都有**INT8的指令集优化**
- 对于某些设备，使用INT8的模型耗电量更少，对于嵌入式侧端设备来说提升是巨大的



## **量化基本知识**

进入主题前需要提两个概念，也就是量化的**两个重要过程**，一个是量化（Quantize），另一个是反量化（Dequantize）：

- 量化就是将浮点型实数量化为整型数（FP32->INT8）
- 反量化就是将整型数转换为浮点型实数（INT8->FP32）

### 量化操作



![image-20210904165740422](D:\notebook\post\机器学习\img\image-20210904165740422.png)



### **基于线性量化的对称量化和非对称量化**

![img](D:\notebook\post\机器学习\img\v2-97354bedca06e959cd19bcab197118e2_b.jpg)

左边是非对称量化 Affine quantization  右边是对称量化 Scale quantization

- 对称量化的实数0也对应着整数的0，而非对称量化的实数0不一定对应着整数0，而是z。
  
- 对称量化实数的范围是对称的 $[-\alpha，\alpha]$ ，而非对称量化的则不对称 $[-\beta，\alpha]$ 

- 对称量化整数的范围是对称的（[-127,127]），而非对称量化的则不对称（[-128,127]）



非对称量化公式可表示为$f(x) = s*x + z$, z 表示实数0映射到整数是多少，对称量化公式表示为$f(x) = s*x $ 



### 对称量化

s量化系数的计算$s = \frac{2^{b-1}-1}{\alpha}$

$x_q = quantize(x,b,s) = clip(round(s\cdot x),-2^{b-1}+1,2^{b-1}-1)$

其中$\alpha$ 代表输入数据分布中的实数最大值，相当于将$0 \sim \alpha$ 之间数据映射到  $0 \sim 127$ 之间

在反量化的时候 $\hat{x} = dequantize(x_q,s) = \frac{x_q}{s}$

那么实际操作过程中，scale系数是怎么用呢？或者说![[公式]](https://www.zhihu.com/equation?tex=s)这个量化系数是怎么作用于所有的输入、所有的权重呢？



一般量化过程中，有`pre-tensor`和`pre-channel`两种方式，`pre-tensor`显而易见，就是对于同一块输入（比如某个卷积前的输入tensor）我们采用一个scale，该层所有的输入数据共享一个scale值；而`pre-channel`呢一般是作用于权重，比如一个卷积的权重维度是[64,3,3,3]（输入通道为3输出通道为64，卷积核为3x3），`pre-channel`就是会产生64个scale值，分别作用于该卷积权重参数的64个通道。

为什么权重不能是`pre-tensor`呢？这个对精度的影响太大了，所以一般不用。输入就可以`pre-tensor`？当然可以，也经过测试了，对精度的影响不是很大，完全可以用。

那为什么权重必须是`pre-channel`呢？不能是每个权重值都有自己的scale么？呃，这个问题嘛，首先可以想到，这个计算量，应该挺大，其次嘛，让我们分析一下。

### **卷积操作量化**

卷积中最重要的就是矩阵相乘

![image-20210904220123260](img\image-20210904220123260.png)

注意看上图输入X的维度为[m,p]而W的维度为[p,n]，因此i的范围为[0,m)，k的范围为[0,p)。W和Y同理。这里的输入和权重都是FP32精度，也就是实数。

$X_q = (x_{q,ik}),W_q = (w_{q,kj})$

$w_{i,j}=\sum_{k=1}^{p}x_{ik} \cdot w_{k,j} \approx \sum_{k=1}^{p} dequantize(x_{q,ik},s_{q,ik}) \cdot dequantize(w_{q,kj},s_{w,kj}) = \sum_{k=1}^{p} \frac{x_{q,ik}}{s_{x,ik}} \cdot  \frac{w_{q,kj}}{s_{x,kj}}$

 从公式中可以看出 x,w 分别有一个scale，当把scale元素提出来，这样x的每一行必须共享scale, w的每一列必须共享scale



#### 卷积 im2col

![图片](img\640wgwrjgpwrgjw)



多通道的im2col的过程，是首先im2col第一通道，然后在im2col第二通道，最后im2col第三通道。需要注意各通道im2col的数据在内存中也是连续存储的，全部弄好后拼成这样的矩阵！

![图片](img\640wgrejgpwrbnwpbnprbnp)

![图片](img\640vsjdjgpwgjwprgjwrgnw;rgnw)

