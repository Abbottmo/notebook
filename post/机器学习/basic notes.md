# 基础知识

# pytorch中交叉熵损失(nn.CrossEntropyLoss())的计算过程详解

```python
## 一维计算
import torch
import torch.nn as nn
import math

criterion = nn.CrossEntropyLoss()
output = torch.randn(1, 5, requires_grad=True)
label = torch.empty(1, dtype=torch.long).random_(5)
loss = criterion(output, label)

print("网络输出为5类:")
print(output)
print("要计算label的类别:")
print(label)
print("计算loss的结果:")
print(loss)

first = 0
for i in range(1):
  first = -output[i][label[i]]
second = 0
for i in range(1):
  for j in range(5):
    second += math.exp(output[i][j])
res = 0
res = (first + math.log(second))
print("自己的计算结果：")
print(res)

##多维计算
import torch
import torch.nn as nn
import math
criterion = nn.CrossEntropyLoss()
output = torch.randn(3, 5, requires_grad=True)
label = torch.empty(3, dtype=torch.long).random_(5)
loss = criterion(output, label)

print("网络输出为3个5类:")
print(output)
print("要计算loss的类别:")
print(label)
print("计算loss的结果:")
print(loss)

first = [0, 0, 0]
for i in range(3):
  first[i] = -output[i][label[i]]
second = [0, 0, 0]
for i in range(3):
  for j in range(5):
    second[i] += math.exp(output[i][j])
res = 0
for i in range(3):
  res += (first[i] + math.log(second[i]))
print("自己的计算结果：")
print(res/3)





```

CrossEntropyLoss计算公式为：
$$
loss(x,class) = -log(\frac{exp(x[class])}{\sum_{j}{exp(x[j])}}) = - x[class] + log(\sum_{j}{exp(x[j])})
$$
CrossEntropyLoss带权重的计算公式为（默认weight=None）：
$$
loss(x,class) = weight[class](- x[class] + log(\sum_{j}{exp(x[j])})) 
$$
