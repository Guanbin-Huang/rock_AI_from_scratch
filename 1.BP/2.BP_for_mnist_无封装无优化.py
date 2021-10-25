""" 
今日知识点预报
    - mnist
    - softmax
    - 逻辑回归是怎么到bp的
    - bp框架
 """
# 数据加载和预处理
import numpy as np
import struct
import random
import matplotlib.pyplot as plt
import pandas as pd 
import math
from my_utils import save_var_to_pkl, load_var_from_pkl


def one_func_set_all_random_seed(seed=0):
    # different random seeds
    import torch
    torch.manual_seed(seed)

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    torch.use_deterministic_algorithms(True)

    # for dataloader
    g = torch.Generator()
    g.manual_seed(seed)

    return g

_ = one_func_set_all_random_seed()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def load_labels(file):
    '''
    解码标签文件
    '''
    with open(file, "rb") as f:
        data = f.read()  # kp: 任务整理 readlines  readline  and read 的区别
    
    magic_number, num_samples = struct.unpack(">ii", data[:8])  # refer to magic_number.jpg  # struct.unpack refer to https://docs.python.org/3/library/struct.html
                                                                # >ii refer to https://docs.python.org/3/library/struct.html
    if magic_number != 2049:
        print(f"magic number mismatch {magic_number} != 2049")
        return None

    labels = np.array(list(data[8:])) # np.asarray  
    return labels

def load_images(file):
    with open(file, "rb") as f: # note rb or r
        data = f.read()
    
    magic_number, num_samples, image_height, image_width = struct.unpack(">iiii", data[:16])

    if magic_number != 2051:
        print(f"magic number mismatch {magic_number} != 2051")
        return None
    
    image_data = np.array(list(data[16:]), dtype=np.uint8).reshape(num_samples, -1) # dtype = "uint8"

    return image_data


train_labels = load_labels("dataset/train-labels-idx1-ubyte")
train_images = load_images("dataset/train-images-idx3-ubyte")
train_numdata = train_labels.shape[0] # 60000
train_pd = pd.DataFrame(train_labels, columns = ["label"])

val_labels = load_labels("dataset/t10k-labels-idx1-ubyte") # 10000
val_images = load_images("dataset/t10k-images-idx3-ubyte") # 10000, 784
val_images = val_images / 255.0
val_numdata = val_labels.shape[0]    # 10000

def one_hot(labels, classes):
    n = len(labels)
    output = np.zeros((n, classes), dtype = np.int32)
    for row, label in enumerate(labels):
        output[row, label] = 1
    return output

def show_hist(labels, num_classes): # 常用的小工具函数的写法
    label_map = {key: 0 for key in range(num_classes)} # 给每一个类都初始化： 数量为0
    for label in labels:       # 循环labels，遇到label x  就去label x的keyvalue对里+1
        label_map[label] += 1  # 这里相当于是一个一个label item去算
    
    # label_hist 是一个list, list 的值是 label_map key-value 对儿里的 value
    labels_hist = [label_map[key] for key in range(num_classes)]  
    pd.DataFrame(labels_hist, columns=["label"]).plot(kind = "bar") # api 用法的形象记忆 refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
                                                                    # refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
# 创建管理数据和数据加载的类
class Dataset:
    # 动态的，那么Dataset是个基类，所有动态的继承自Dataset
    # 需要实现什么接口？
    def __getitem__(self, index):
        raise NotImplementedError()
        
    def __len__(self):
        raise NotImplementedError()

class MNIST_Dataset(Dataset):
    # 针对mnist数据的解析、加载、预处理(e.g. /255), 加一个全是1的维度etc
    def __init__(self, image_file, label_file):
        self.num_classes = 10
        self.images = load_images(image_file)
        self.labels = load_labels(label_file)

        # self.images = np.hstack((self.images / 255.0, np.ones((len(self.images), 1)))).astype(np.float32)
        self.images = (self.images / 255.0).astype(np.float64) # 64
        self.labels_one_hot = one_hot(self.labels, self.num_classes)
        
    def __getitem__(self, index):
        """ 
        角色的职责
        实现图像加载、归一化/标准化、onehot
            为什么要返回one_hot，计算时，使用one_hot比较方便
            为什么要返回label，因为做测试的时候，label比较方便
            pytorch里面，CELoss使用的不是one_hot。所以以后不需要返回one_hot
         """
        return self.images[index], self.labels[index], self.labels_one_hot[index]

    # 获取数据集的长度，个数
    def __len__(self):
        return len(self.images)

class DataLoader:
    """
    职责
    实例化的时候需要指定dataset，batch_size，shuffle
    数据的封装，打包为一个batch
    对数据进行打乱
    可以通过迭代器来获取一批一批的数据
     """
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.shuffle = shuffle
        self.count_data = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        # 实例化一个迭代器对象，将自身作为参数传入进去
        return DataLoaderIterator(self)

    def __len__(self):
        """ 
        用以告诉外界，多少次迭代，就算是完成一轮
        这里有2种处理方法
        1.向上取整
        2.整除，向下取整，多余部分丢弃
        这里考虑用策略2
         """
        return len(self.dataset) // self.batch_size
        
class DataLoaderIterator:
    """ 
    职责：
        对打包好的batch一个一个的输出
     """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        
        # 这里有2中处理策略
        # 1.向上取整
        # 2.整除，向下取整，多余部分丢弃
        # 这里考虑用方法2
        self.num_batch_per_epoch = len(dataloader)
        
        # 定义指针记录当前batch的索引
        self.batch_cursor = 0

        # 实现一轮数据的打乱和封装获取
        # 与其打乱数据，不如打乱索引
        self.indexes = list(range(len(dataloader.dataset)))

        # 如果需要随机打乱，条件控制由dataloader的shuffle决定
        if dataloader.shuffle:
            np.random.shuffle(self.indexes)  # inplace e.g. [0,1,2,3 ....59999] --> [2,1,48,23,...0]
    
    def __next__(self): # 指的是next batch
        # 如果到了一轮的边界，即迭代结束，抛出异常 (一上来就做判断)
        if self.batch_cursor >= self.num_batch_per_epoch:
            # 如果到了边界，抛出StopIteration
            raise StopIteration()
        """ 
        职责：如何一个又一个的数据进行吐出, 每一行是一个数据
            b1  image.shape = 784,     label.shape = 1,     label_onehot.shape = 10,
            b2  image.shape = 784,     label.shape = 1,     label_onehot.shape = 10,
            b3  image.shape = 784,     label.shape = 1,     label_onehot.shape = 10,
            ......
            n 个 data
        
        images.shape = n x 784     labels.shape = n x 1        one_hot.shape = n x 10
         """ 

        batch_data = []
        for i in range(self.dataloader.batch_size): # 遍历一个batch里的图片
            """ 
             拿到图像的索引，这个索引可能是打乱的
              """
            index = self.indexes[self.batch_cursor * self.dataloader.batch_size + i] # 全局idx
            # 从dataset中拿到数据 e.g. 一个数据由图像和标签组成
            data_item = self.dataloader.dataset[index]

            if len(batch_data) == 0:
                batch_data = [[] for _ in data_item] # 这里有3个
            
            # 把data_item中的每一项，分门别类的放到batch_data中
            for index, item in enumerate(data_item):
                batch_data[index].append(item)


        # 遍历完了这个batch里的所有图片，要到下一个batch了
        self.batch_cursor += 1

        # 当整个batch的数据准备好过后，可以用np.vstack拼接在一起
        for index in range(len(batch_data)):
            batch_data[index] = np.vstack(batch_data[index])

        return tuple(batch_data)

# 初始化权重，和定义网络
num_train_images = train_images.shape[0]
num_feature = train_images.shape[1]
num_hidden = 1024
num_classes = 10
batch_size = 256
lr = 0.1
num_epochs = 10

# 加载数据
train_dataset = MNIST_Dataset("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte")
train_loader  = DataLoader(train_dataset, batch_size, True)
test_dataset = MNIST_Dataset("dataset/t10k-images-idx3-ubyte", "dataset/t10k-labels-idx1-ubyte")
test_loader  = DataLoader(test_dataset, 500, False)


# 初始化权重
W = np.random.normal(0, 1, size = (num_feature, num_hidden)) # (785, num_hidden)
b1 = np.zeros((1, num_hidden))

U = np.random.normal(0, 1, size = (num_hidden, num_classes)) # (num_hidden, 10)
b2 = np.zeros((1, num_classes))

print(f"start training: \nlr:{lr}  batchsize {batch_size}  num_hidden: {num_hidden}")

for epoch in range(num_epochs):
    for images, labels, onehot_labels in train_loader:

        # ######################## FORWARD
        # layer1
        Hz = images @ W + b1
        Ha = sigmoid(Hz)

        # layer2
        output = Ha @ U + b2

        # Softmax Cross Entropy Loss 计算
        probability = softmax(output)
        loss = -np.sum(onehot_labels * np.log(probability)) / batch_size

        # ######################## BACKWARD
        G = (probability - onehot_labels) / batch_size # del_predict 
        
        del_U = Ha.T @ G
        del_b2 = np.sum(G)
        del_Ha = G @ U.T
    
        del_Hz = del_Ha * sigmoid(Hz) * (1 - sigmoid(Hz))
        
        del_W = images.T @ del_Hz
        
        del_b1 = np.sum(del_Hz)
        
        #region gradient checking
        # input -> W -> U -> output  del_W : dLoss/dw  
        if False:
            new_W = W.copy()
            eps = 1e-5
            new_W[100,100] += eps

            Hz = images @ new_W + b1
            Ha = sigmoid(Hz)

            output = Ha @ U + b2

            # Softmax Cross Entropy Loss 计算
            probability = softmax(output)
            new_loss = -np.sum(onehot_labels * np.log(probability)) / batch_size
            
            dloss = new_loss - loss
            del_W_checking = dloss / eps
            
            diff = del_W_checking - del_W[100,100]
            
            print(f"误差：{diff}")

        #endregion
        # ####################### UPDATE
        U -= lr * del_U
        b2 -= lr * del_b2
        W -= lr * del_W
        b1 -= lr * del_b1

    correct = 0
    for test_images, test_labels, test_one_hot_labels in test_loader:
        Ha = sigmoid(test_images @ W + b1)
        probability = softmax(Ha @ U + b2)
        predict_labels     = probability.argmax(axis=1).reshape(-1, 1)
        correct          += (predict_labels == test_labels).sum()
    
    acc = correct / len(test_dataset)
    print(f"{epoch}. train_Loss: {loss:.3f}, test_Accuracy: {acc:.5f}")       
