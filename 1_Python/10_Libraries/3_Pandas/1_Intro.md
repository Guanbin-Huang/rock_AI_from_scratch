&emsp;
# Pandas 


相关链接
- Pandas 官网 https://pandas.pydata.org/
- Pandas 源代码：https://github.com/pandas-dev/pandas

安装Pandas（Linux）
```python
~/anaconda3/envs/[自己的环境名]/bin/pip install pandas
```

&emsp;
# 1 介绍
- Pandas 是 Python 语言的一个扩展程序库，用于数据分析。

- Pandas 是一个开放源码、BSD 许可的库，提供高性能、易于使用的数据结构和数据分析工具。

- Pandas 名字衍生自术语 "panel data"（面板数据）和 "Python data analysis"（Python 数据分析）。

- Pandas 一个强大的分析结构化数据的工具集，基础是 Numpy（提供高性能的矩阵运算）。

- Pandas 可以从各种文件格式比如 CSV、JSON、SQL、Microsoft Excel 导入数据。

- Pandas 可以对各种数据进行运算操作，比如归并、再成形、选择，还有数据清洗和数据加工特征。

- Pandas 广泛应用在学术、金融、统计学等各个数据分析领域。

&emsp;
# 2 Pandas 应用
Pandas 的主要数据结构是 Series （一维数据）与 DataFrame（二维数据），这两种数据结构足以处理金融、统计、社会科学、工程等领域里的大多数典型用例。

&emsp;
# 3 数据结构
Series 是一种类似于一维数组的对象，它由一组数据（各种Numpy数据类型）以及一组与之相关的数据标签（即索引）组成。

DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。
