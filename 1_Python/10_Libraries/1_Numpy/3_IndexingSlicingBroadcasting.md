
&emsp;
# NumPy 索引、切片和广播

&emsp;
# 1 索引和切片
>格式
```python
ndarray[start:stop:step]
```
ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。

>示例
```python
import numpy as np

a = np.arange(10)  
b = a[2:7:2]   # 从索引 2 开始到索引 7 停止，间隔为 2
print(b)
'''输出结果为：
[2  4  6]'''

a = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
b = a[5] 
print(b)
'''输出结果为：
5'''

a = np.arange(10)
print(a[2:])
'''输出结果为：
[2  3  4  5  6  7  8  9]'''

a = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
print(a[2:5])
'''输出结果为：
[2  3  4]'''

# 多维数组同样适用上述索引提取方法：
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
# 从某个索引处开始切割
print('从数组索引 a[1:] 处开始切割')
print(a[1:])


# 切片还可以包括省略号 ...，来使选择元组的长度与数组的维度相同。 如果在行位置使用省略号，它将返回包含行中元素的 ndarray。
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print (a[...,1])   # 第2列元素
print (a[1,...])   # 第2行元素
print (a[...,1:])  # 第2列及剩下的所有元素
'''输出结果为：
[2 4 5]
[3 4 5]
[[2 3]
 [4 5]
 [5 6]]'''
```

&emsp;
# 2 NumPy 高级索引
数组可以由整数数组索引、布尔索引及花式索引。

&emsp;
## 2.1 整数数组索引
以下实例获取数组中(0,0)，(1,1)和(2,0)位置处的元素。

>示例
```python
import numpy as np 

# 示例1：
x = np.array([[1,  2],  [3,  4],  [5,  6]]) 
y = x[[0,1,2],  [0,1,0]]  # 分别代表行索引和列索引，等价于:
                          # [x[0][0], x[1][1], x[2][0]]
print(x)
print("-----------")
print(y)
'''输出结果为：
[1  4  5]'''

# 示例2：
x = np.array([[0, 1, 2],[3, 4, 5],[6, 7, 8],[9, 10, 11]])  
print (x)
rows = np.array([[0,0],[3,3]]) 
cols = np.array([[0,2],[0,2]]) 
print("rows: \n", rows)
print("cols: \n", cols)

y = x[rows,cols]  
print  ('这个数组的四个角元素是：')
print (y)

# 示例3：可以借助切片 : 或 … 与索引数组组合
a = np.array([[1,2,3], [4,5,6],[7,8,9]])
b = a[1:3, 1:3]
c = a[1:3,[1,2]]
d = a[...,1:]
print(b)
print(c)
print(d)
```


&emsp;
## 2.2 布尔索引
- 我们可以通过一个布尔数组来索引目标数组。
- 布尔索引通过布尔运算（如：比较运算符）来获取符合指定条件的元素的数组。

>示例
```python
import numpy as np 

# 示例1：获取大于 5 的元素：
x = np.array([[0, 1, 2],[3, 4, 5],[6, 7, 8],[9, 10, 11]])  
print ('我们的数组是：')
print (x)
print  ('大于 5 的元素是：')
print (x[x > 5])


# 示例2：从数组中过滤掉非复数元素。
a = np.array([1,  2+6j,  5,  3.5+5j])  
mask = np.iscomplex(a)
print (a[mask])
'''输出如下：
[2.0+6.j  3.5+5.j]'''
```

&emsp;
## 2.3 花式索引
- 花式索引指的是利用整数数组进行索引。
- 花式索引根据索引数组的值作为目标数组的某个轴的下标来取值。
- 花式索引跟切片不一样，它总是将数据复制到新数组中。

&emsp;
### (1) 传入顺序索引数组
>示例
```python
import numpy as np 
 
x=np.arange(32).reshape((8,4))
print(x)
print("---------")
print(x[[4,2,1,7]]) # 分别取 第4、2、1、7行作为新数组返回
```

&emsp;
### (2) 传入倒序索引数组

>示例
```python
import numpy as np 
x=np.arange(32).reshape((8,4))
print (x[[-4,-2,-1,-7]])
'''输出结果为：
[[16 17 18 19]
 [24 25 26 27]
 [28 29 30 31]
 [ 4  5  6  7]]'''
```

&emsp;
# 3 NumPy 广播(Broadcast)
广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式。

- 广播的条件
    - (1) 数组维数不相等，但后缘维度的轴长相等
    - (2) 后缘维度有一方以上长度为1，其它相等

>示例
```python
import numpy as np 

# 示例1：后缘3个维度相等
a = np.arange(48).reshape(2, 3, 4, 2)
b = np.arange(8).reshape(4, 2)
print(a)
print("-----------------")
print(b)
print("-----------------")
print(a + b)

# 示例2：后缘2个维度相等
a = np.arange(48).reshape(2, 3, 4, 2)
b = np.arange(8).reshape(4, 2)
print(a)
print("-----------------")
print(b)
print("-----------------")
print(a + b)

# 示例3：后缘3个维度, 其中一个为1，其它相等
a = np.arange(48).reshape(2, 3, 4, 2)
b = np.arange(6).reshape(3, 1, 2)
print(a)
print("-----------------")
print(b)
print("-----------------")
print(a + b)

```


