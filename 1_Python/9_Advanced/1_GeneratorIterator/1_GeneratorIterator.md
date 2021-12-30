&emsp;
# 迭代器与生成器

&emsp;
# 1 迭代器
- 迭代是Python最强大的功能之一，是访问集合元素的一种方式。

- 迭代器是一个可以记住遍历的位置的对象。

- 迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。

- 迭代器有两个基本的方法：iter() 和 next()。

- 字符串，列表或元组对象都可用于创建迭代器：

>示例
```python
list=[1,2,3,4]
it = iter(list)    # 创建迭代器对象
print (next(it))   # 输出迭代器的下一个元素
# 输出：1
print (next(it))
# 输出：2
```

迭代器对象可以使用常规for语句进行遍历：

>示例
```python
list=[1,2,3,4]
it = iter(list)    # 创建迭代器对象
for x in it:
    print (x, end = " ")
```


也可以使用 next() 函数：

>示例
```python
import sys         # 引入 sys 模块
list=[1,2,3,4]
it = iter(list)    # 创建迭代器对象
 
while True:
    try:
        print (next(it))
    except StopIteration:
        sys.exit()
```

&emsp;
## 1.1 创建一个迭代器
- 把一个类作为一个迭代器使用需要在类中实现两个方法 \_\_iter__() 与 \_\_next__() 。

    - `__iter__()` 方法返回一个特殊的迭代器对象， 这个迭代器对象实现了 
    - `__next__()` 方法并通过 StopIteration 异常标识迭代的完成。

```python
class Dataset():
    def __init__(self, images, labels):
        self.images = mnist_images(images)
        self.labels = mnist_labels(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index], self.images[index]

class Loader():
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor > len(self.dataset):
            raise StopIteration
        data = self.dataset[self.cursor]
        self.cursor += 1
        return data
```


&emsp;
## 1.2 StopIteration
&emsp;&emsp;StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况，在 \_\_next__() 方法中我们可以设置在完成指定循环次数后触发 StopIteration 异常来结束迭代。


>示例
```python
from utils import *

class Dataset():
    def __init__(self, images, labels):
        self.images = mnist_images(images)
        self.labels = mnist_labels(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index], self.images[index].shape


class DataLoader():
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return DataLoaderIterator(self.dataset)

class DataLoaderIterator():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.cursor = 0

    def __next__(self):
        if self.cursor > len(self.dataset):
            raise StopIteration
        self.cursor += 1
        return self.dataset[self.cursor]


if __name__ == '__main__':
    test_label = "/datav/MyLesson/Dataset/MNIST/raw/t10k-labels-idx1-ubyte"
    test_images = "/datav/MyLesson/Dataset/MNIST/raw/t10k-images-idx3-ubyte"

    test_set = Dataset(test_images, test_label)
    # print(test_set[0]) # 7
    # print(test_set[1]) # 2
    it = iter(test_set)
    print(next(it))
    print(next(it))
```

&emsp;
# 2 生成器
- 在 Python 中，使用了 `yield` 的函数被称为生成器（generator）。

- 跟普通函数不同的是，生成器是一个返回迭代器的函数，只能用于迭代操作，更简单点理解生成器就是一个迭代器。

- 在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值, 并在下一次执行 next() 方法时从当前位置继续运行。

- 调用一个生成器函数，返回的是一个迭代器对象。

>示例
```python
from utils import *

class Dataset():
    def __init__(self, images, labels):
        self.images = mnist_images(images)
        self.labels = mnist_labels(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index], self.images[index].shape

def DataLoader(dataset):
    cursor = 0
    while True:
        if cursor > len(dataset):
            raise StopIteration
        label = dataset.labels[cursor]
        image = dataset.images[cursor]
        cursor += 1

        yield label, image


if __name__ == '__main__':
    test_label = "/datav/MyLesson/Dataset/MNIST/raw/t10k-labels-idx1-ubyte"
    test_images = "/datav/MyLesson/Dataset/MNIST/raw/t10k-images-idx3-ubyte"

    test_set = Dataset(test_images, test_label)
    # print(test_set[0]) # 7
    # print(test_set[1]) # 2
    test_loader = DataLoader(test_set)
    count = 0
    for label, image in it:
        if count > 1:
            break
        print(label, image.shape)
        count += 1
```

3 enumerate
4 zip