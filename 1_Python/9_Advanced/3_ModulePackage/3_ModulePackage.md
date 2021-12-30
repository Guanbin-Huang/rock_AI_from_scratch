&emsp;
# Module模块

- 一个包含定义的函数和变量的文件
- 后缀名是 .py
- 模块可以被别的程序引入，以使用该模块中的函数等功能。这也是使用 python 标准库的方法。

>示例
```python
import sys 

for i in sys.argv:
    print(i)

print(sys.path)
```
- import sys 引入 python 标准库中的 sys.py 模块；这是引入某一模块的方法。
- sys.argv 是一个包含命令行参数的列表。
- sys.path 包含了一个 Python 解释器自动查找所需模块的路径的列表。

&emsp;
# 1 import 语句

&emsp;
## 1.1 基本使用
- 导入准备使用的 Python 源文件
>语法
```python
import module1[, module2[,... moduleN]
```
- 当解释器遇到 import 语句，如果模块在当前的搜索路径就会被导入。
- 搜索路径是一个解释器会先进行搜索的所有目录的列表。

>support.py文件代码
```python
def print_func( par ):
    print ("Hello : ", par)
    return
```

>test.py文件代码
```python
import support # test.py 引入 support 模块
support.print_func("AAAA")

# 可以给引入的模块中函数起别名
tt = support.print_func
tt("BBBBB")
```

- 不管你执行了多少次import，一个模块只会被导入一次。这样可以防止导入模块被一遍又一遍地执行。

- Python 中 import 语句的搜索路径，搜索路径是由一系列目录名（sys.path）组成的，Python解释器就依次从这些目录中去寻找所引入的模块。这看起来很像环境变量，也可以通过定义环境变量的方式来确定搜索路径。

- 搜索路径是在Python编译或安装的时候确定的，安装新的库应该也会修改。搜索路径被存储在sys模块中的path变量


&emsp;
# 2 深入模块

&emsp;&emsp;每个模块有各自独立的符号表，在模块内部为所有的函数当作全局符号表来使用。所以，模块的作者可以放心大胆的在模块内部使用这些全局变量，而不用担心把其他用户的全局变量搞混。

&emsp;&emsp;从另一个方面，当你确实知道你在做什么的话，你也可以通过 modname.itemname 这样的表示法来访问模块内的函数。

&emsp;&emsp;模块是可以导入其他模块的。在一个模块（或者脚本，或者其他地方）的最前面使用 import 来导入一个模块。被导入的模块的名称将被放入当前操作的模块的符号表中。

&emsp;
## 2.1 from … import 语句
- Python 的 from 语句让你从模块中导入一个指定的部分到当前命名空间中
>语法
```python
from modname import name1[, name2[, ... nameN]]
```
例如，要导入模块 A 的 a, b 函数，使用如下语句：
```python
from A import a, b
```
- 这个声明不会把整个 A 模块导入到当前的命名空间中，它只会将 A 里的a, b函数引入进来。



&emsp;
## 2.2 from … import * 语句
把一个模块的所有内容全都导入到当前的命名空间也是可以的
>语法
```python
from modname import *
```
- 这提供了一个简单的方法来导入一个模块中的所有项目。
- 可以一次性的把模块中的所有（函数，变量）名称都导入到当前模块的字符表，但是那些由单一下划线（_）开头的名字不在此例。
- 大多数情况， Python 程序员不使用这种方法，因为引入的其它来源的命名，很可能覆盖了已有的定义。

&emsp;
## 2.3 __name__属性
&emsp;&emsp;一个模块被另一个程序第一次引入时，其主程序将运行。如果我们想在模块被引入时，模块中的某一程序块不执行，我们可以用__name__属性来使该程序块仅在该模块自身运行时执行。

>support.py文件
```python
print("不在__name__ == '__main__'中")

if __name__ == '__main__':
    print("__name__ == '__main__'中执行")
```
>test.py文件
```python
import support
```

- 说明： 
    - 每个模块都有一个__name__属性，当其值是'\_\_main__'时，表明该模块自身在运行，否则是被引入。

&emsp;
## 2.4 dir() 函数
- 内置的函数 dir() 可以找到模块内定义的所有名称。以一个字符串列表的形式返回
- 如果没有给定参数，那么 dir() 函数会罗列出当前定义的所有名称

>support.py文件
```python
a = [1, 2, 3, 4, 5]

```


>test.py文件
```python
import support

print(dir(support))
print(dir()) # 得到一个当前模块中定义的属性列表
xxx = 5
print(dir()) # 新增了 "xxx"
del xxx
print(dir()) # 删除了 "xxx"
```


&emsp;
# 3 包
&emsp;&emsp;包是一种管理 Python 模块命名空间的形式，采用"点模块名称"。

&emsp;&emsp;比如一个模块的名称是 A.B， 那么他表示一个包 A 中的子模块 B 。

&emsp;&emsp;就好像使用模块的时候，你不用担心不同模块之间的全局变量相互影响一样，采用点模块名称这种形式也不用担心不同库之间的模块重名的情况。

>示例

- 假设有一套统一处理声音文件和数据的模块（或者称之为一个"包"）。
- 音频文件格式有： .wav，:file:.aiff，:file:.au等，所以你需要有一组不断增加的模块，用来在不同的格式之间转换。
- 针对这些音频数据，有很多操作（比如混音，添加回声，增加均衡器功能，创建人造立体声效果），所以你还需要一组怎么也写不完的模块来处理这些操作。
- 包结构（在分层的文件系统中）大概如下:

```
sound/                          顶层包
      __init__.py               初始化 sound 包
      formats/                  文件格式转换子包
              __init__.py
              wavread.py
              wavwrite.py
              aiffread.py
              aiffwrite.py
              auread.py
              auwrite.py
              ...
      effects/                  声音效果子包
              __init__.py
              echo.py
              surround.py
              reverse.py
              ...
      filters/                  filters 子包
              __init__.py
              equalizer.py
              vocoder.py
              karaoke.py
              ...
```

&emsp;&emsp;在导入一个包的时候，Python 会根据 sys.path 中的目录来寻找这个包中包含的子目录。

&emsp;&emsp;目录只有包含一个叫做 \_\_init__.py 的文件才会被认作是一个包，主要是为了避免一些滥俗的名字（比如叫做 string）不小心的影响搜索路径中的有效模块。

&emsp;&emsp;最简单的情况，放一个空的 :file:\_\_init__.py就可以了。当然这个文件中也可以包含一些初始化代码或者为 __all__变量赋值。

>示例1

用户可以每次只导入一个包里面的特定模块，比如:
```python
import sound.effects.echo
# 这将会导入子模块:sound.effects.echo。 他必须使用全名去访问:
sound.effects.echo.echofilter(input, output, delay=0.7, atten=4)
```

>示例2

还有一种导入子模块的方法是:
```python
from sound.effects import echo
# 这同样会导入子模块: echo，并且他不需要那些冗长的前缀，所以他可以这样使用:
echo.echofilter(input, output, delay=0.7, atten=4)
```

>示例3
还有一种变化就是直接导入一个函数或者变量:

```python
from sound.effects.echo import echofilter
#同样的，这种方法会导入子模块: echo，并且可以直接使用他的 echofilter() 函数:
echofilter(input, output, delay=0.7, atten=4)
```

>注意
- 使用 from package import item 时，item 既可以是包里面的子模块（子包），或者包里面定义的其他名称，比如函数，类或者变量。import 语法会首先把 item 当作一个包定义的名称，如果没找到，再试图按照一个模块去导入。如果还没找到，抛出一个 :exc:ImportError 异常。

- 如果使用 import item.subitem.subsubitem 这种导入形式，除了最后一项，都必须是包，而最后一项则可以是模块或者是包，但是不可以是类，函数或者变量的名字。

&emsp;
# 4 从一个包中导入*

&emsp;
## 4.1 \_\_all__
如果我们使用 from sound.effects import * 会发生什么呢？
- Python 会进入文件系统，找到这个包里面所有的子模块，然后一个一个的把它们都导入进来 

- 如果包定义文件 `__init__.py` 存在一个叫做 `__all__` 的列表变量，那么在使用 from package import * 的时候就把这个列表中的所有名字作为包内容导入 

- 在更新包函数和变量等内容后，记住 `__all__` 也要更新

>sounds/effects/\_\_init__.py 中代码
```python
__all__ = ["echo", "surround", "reverse"]
# 这表示当你使用from sound.effects import * 这种用法时，你只会导入包里面这三个子模块。
```

如果 `__all__` 没有定义，使用from sound.effects import *就不会导入包 sound.effects 里的任何子模块。他只是把包sound.effects和它里面定义的所有内容导入进来（可能运行__init__.py里定义的初始化代码）。

这会把 \_\_init__.py 里面定义的所有名字导入进来。并且他不会破坏掉我们在这句话之前导入的所有明确指定的模块。看下这部分代码:
```python
import sound.effects.echo
import sound.effects.surround
from sound.effects import *
```
这个例子中，在执行 from...import 前，包 sound.effects 中的 echo 和 surround 模块都被导入到当前的命名空间中了。（当然如果定义了 \_\_all__ 就更没问题了）

&emsp;
## 4.2 总结
（1）不推荐使用 * 这种方法来导入模块，因为这种方法经常会导致代码的可读性降低。不过这样倒的确是可以省去不少敲键的功夫

（2）使用 from Package import specific_submodule 这种方法永远不会有错。这也是推荐的方法。除非是你要导入的子模块有可能和其他包的子模块重名。

