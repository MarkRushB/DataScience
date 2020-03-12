# The charm of Data Science
In order to complete this document, I combined self-study materials and NEU course `INFO6105-Data Sci Eng Methods` materials. 
## Introduce to Python
[Part-1](./DataScience%20Reference/Introduction-to-Python-sp20-start.ipynb)
[Part-2](./DataScience%20Reference/Introduction-to-Python-sp20-part2-start.ipynb)
## Numpy
### Numpy简介
Numpy是一个用python实现的科学计算的扩展程序库，包括：
1. 一个强大的N维数组对象Array；
2. 比较成熟的（广播）函数库；
3. 用于整合C/C++和Fortran代码的工具包；
4. 实用的线性代数、傅里叶变换和随机数生成函数。numpy和稀疏矩阵运算包scipy配合使用更加方便。

NumPy（Numeric Python）提供了许多高级的数值编程工具，如：矩阵数据类型、矢量处理，以及精密的运算库。专为进行严格的数字处理而产生。多为很多大型金融公司使用，以及核心的科学计算组织如：Lawrence Livermore，NASA用其处理一些本来使用C++，Fortran或Matlab等所做的任务。

**导入：**
```python
import numpy as np
```

### Array Attributes
We'll start by defining three random arrays, a one-dimensional, two-dimensional, and three-dimensional array.

We'll use NumPy's random number generator, which we will *seed* with a set value in order to ensure that the same random arrays are generated each time this code is run:
## 

