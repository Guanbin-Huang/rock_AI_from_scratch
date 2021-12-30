&emsp;
# Python 输入和输出

&emsp;
# 1 读和写文件
open() 将会返回一个 file 对象，基本语法格式如下:
```python
open(filename, mode)
```
- filename：包含了你要访问的文件名称的字符串值。
- mode：决定了打开文件的模式：只读，写入，追加等。所有可取值见如下的完全列表。这个参数是非强制的，默认文件访问模式为只读(r)。

不同模式打开文件的完全列表：

模式	|描述
:--|:--
r	|以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。
rb	|以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。
r+	|打开一个文件用于读写。文件指针将会放在文件的开头。
rb+	|以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。
w	|打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
wb	|以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
w+	|打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
wb+	|以二进制格式打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
a	|打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
ab	|以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
a+	|打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。
ab+	|以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。

下图很好的总结了这几种模式：

模式	|r	|r+	|w	|w+	|a	|a+
:--|:--|:--|:--|:--|:--|:--
读	|+	|+	|	|+	|	|+
写	|	|+	|+	|+	|+	|+
创建	|	|	|+	|+	|+	|+
覆盖	|	|	|+	|+	|	|
指针在开始	|+	|+	|+	|+	|	|
指针在结尾	|	|	|	|	|+	|+

>示例
```python
# 将字符串写入到文件 foo1.txt 中：
# 打开一个文件
f = open("./files/foo1.txt", "w")
f.write( "Python 是一个非常好的语言。\n是的，的确非常好!!\n" )
# 关闭打开的文件
f.close()
```

- 第一个参数为要打开的文件名。
- 第二个参数描述文件如何使用的字符。 mode 可以是 'r' 如果文件只读, 'w' 只用于写 (如果存在同名文件则将被删除), 和 'a' 用于追加文件内容; 所写的任何数据都会被自动增加到末尾. 'r+' 同时用于读写。 mode 参数是可选的; 'r' 将是默认值。

此时打开文件 foo.txt,显示如下：
```shell
$ cat ./files/foo.txt 

Python 是一个非常好的语言。
是的，的确非常好!!
```

&emsp;
# 2 文件对象的方法
本节中剩下的例子假设已经创建了一个称为 f 的文件对象。

&emsp;
## 2.1 f.read()
为了读取一个文件的内容，调用 f.read(size), 这将读取一定数目的数据, 然后作为字符串或字节对象返回。

size 是一个可选的数字类型的参数。 当 size 被忽略了或者为负, 那么该文件的所有内容都将被读取并且返回。


>示例
```python
# 假定文件 foo1.txt 已存在
# 打开一个文件
f = open("./files/foo1.txt", "r")
str = f.read()
print(str)
# 关闭打开的文件
f.close()
'''执行以上程序，输出结果为：
Python 是一个非常好的语言。
是的，的确非常好!!'''
```

&emsp;
## 2.2 f.readline()
- f.readline() 会从文件中读取单独的一行。换行符为 '\n'。f.readline() 如果返回一个空字符串, 说明已经已经读取到最后一行。

>示例
```python
# 打开一个文件
f = open("./files/foo1.txt", "r")
str = f.readline()
print(str)
# 关闭打开的文件
f.close()
'''执行以上程序，输出结果为：
Python 是一个非常好的语言。'''
```

&emsp;
## 2.3 f.readlines()
f.readlines() 将返回该文件中包含的所有行。

如果设置可选参数 sizehint, 则读取指定长度的字节, 并且将这些字节按行分割。

>示例
```python
# 打开一个文件
f = open("./files/foo1.txt", "r")
str = f.readlines()
print(str)
# 关闭打开的文件
f.close()
'''执行以上程序，输出结果为：
['Python 是一个非常好的语言。\n', '是的，的确非常好!!\n']'''

# 另一种方式是迭代一个文件对象然后读取每行
# 打开一个文件
f = open("./files/foo1.txt", "r")
for line in f:
    print(line, end='')
# 关闭打开的文件
f.close()
'''执行以上程序，输出结果为：

Python 是一个非常好的语言。
是的，的确非常好!!'''
```


&emsp;
## 2.4 f.write()
f.write(string) 将 string 写入到文件中, 然后返回写入的字符数。

>示例
```python
# 打开一个文件
f = open("./files/foo2.txt", "w")

num = f.write( "Python 是一个非常好的语言。\n是的，的确非常好!!\n" )
print(num)
# 关闭打开的文件
f.close()


# 如果要写入一些不是字符串的东西, 那么将需要先进行转换:
# 打开一个文件
f = open("./files/foo3.txt", "w")

value = ('www.runoob.com', 14)
s = str(value)
f.write(s)

# 关闭打开的文件
f.close()
'''执行以上程序，打开 foo3.txt 文件：

$ cat ./files//foo3.txt 
('www.runoob.com', 14)'''
```

&emsp;
## 2.5 f.tell()
f.tell() 返回文件对象当前所处的位置, 它是从文件开头开始算起的字节数。


&emsp;
## 2.6 f.seek()
如果要改变文件当前的位置, 可以使用函数
```python
f.seek(offset, from_what)
```
- 字符偏移量，正数代表往后偏移，负数代表往前偏移
- from_what 的值, 如果是 0 表示开头, 如果是 1 表示当前位置, 2 表示文件的结尾，例如：

```python
f = open('./files/foo2.txt', 'rb+')
f.write(b'0123456789abcdef')
f.seek(5)     # 移动到文件的第六个字节
f.read(1)
f.seek(-3, 2) # 移动到文件的倒数第三字节
f.read(1)
f.close()
```

&emsp;
## 2.7 f.close()
在文本文件中 (那些打开文件的模式下没有 b 的), 只会相对于文件起始位置进行定位。

当你处理完一个文件后, 调用 f.close() 来关闭文件并释放系统的资源，如果尝试再调用该文件，则会抛出异常。
```python
>>> f.close()
>>> f.read()
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
ValueError: I/O operation on closed file
```
当处理一个文件对象时, 使用 with 关键字是非常好的方式。在结束后, 它会帮你正确的关闭文件。 而且写起来也比 try - finally 语句块要简短:

```python
>>> with open('./files/foo.txt', 'r') as f:
...     read_data = f.read()
>>> f.closed
True
```

&emsp;
# 3 pickle 模块
- python的pickle模块实现了基本的数据序列和反序列化。
  - pickle模块只能在python中使用，python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
  - pickle序列化后的数据，可读性差，人一般无法识别。
  - 通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储。
  - 通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。

基本接口：
```python
pickle.dump(obj, file, [,protocol])
```
有了 pickle 这个对象, 就能对 file 以读取的形式打开:
```python
x = pickle.load(file)
```
注解：
- 从 file 中读取一个字符串，并将它重构为原来的python对象。
- file: 类文件对象，有read()和readline()接口。
- protocol是序列化模式，默认值为0，表示以文本的形式序列化。protocol的值还可以是1或2，表示以二进制的形式序列化。

```python
import pickle

# 使用pickle模块将数据对象保存到文件
data1 = {'a': [1, 2.0, 3, 4+6j],
         'b': ('string', u'Unicode string'),
         'c': None}

selfref_list = [1, 2, 3]
selfref_list.append(selfref_list)

output = open('./files/data.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(data1, output)

# Pickle the list using the highest protocol available.
pickle.dump(selfref_list, output, -1) #  protocal 的值是负数， 使用最高 protocal 对 obj 压缩

output.close()

#实例 2
import pprint, pickle

#使用pickle模块从文件中重构python对象
pkl_file = open('./files/data.pkl', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

data2 = pickle.load(pkl_file)
pprint.pprint(data2)

pkl_file.close()
```