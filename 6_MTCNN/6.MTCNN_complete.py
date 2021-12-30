""" 
This version allows the user to input one image then show the output.
 """


import numpy as np
import struct
import random
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import isin
import pandas as pd
import math
import pickle as pkl
import time
import torch
import cv2
from my_utils import save_var_to_pkl, load_var_from_pkl

#-------------------------------------------------- 工具函数 ------------------------------------------------------
#region
class BBox:
    '''
    业界规范：BBox[x,y,r,b] xy:左上角xy坐标  rb:右下角xy坐标
    '''

    def __init__(self, x, y, r, b, score=0):
        self.x, self.y, self.r, self.b, self.score = x, y, r, b, score
        self.height = b - y + 1
        self.width = r - x + 1
        self.maxl = max(self.height,self.width)
        self.minl = min(self.height,self.width)

    def __repr__(self):
        return f"{self.x:.2f}, {self.y:.2f}, {self.r:.2f}, {self.b:.2f}, {self.score:.2f}"

    def __and__(self, other):
        '''
        计算box和other的交集
        '''
        x_max = min(self.r, other.r)
        y_max = min(self.b, other.b)
        x_min = max(self.x, other.x)
        y_min = max(self.y, other.y)

        cross_box = BBox(x_min, y_min, x_max, y_max)
        if cross_box.width <= 0 or cross_box.height <= 0:
            return 0

        return cross_box.area

    def __or__(self, other):
        '''
        计算box和other的并集
        '''
        cross = self & other
        union = self.area + other.area - cross

        return union

    def __xor__(self, other):
        cross = self & other
        union = self | other
        return cross / (union + 1e-6)

    def boundof(self, other):
        '''
        计算box和other的边缘外包框，使得2个box都在框内的最小矩形
        '''
        x_min = min(self.x, other.x)
        y_min = min(self.y, other.y)
        x_max = max(self.r, other.r)
        y_max = max(self.b, other.b)

        return BBox(x_min, y_min, x_max, y_max)

    def center_dist(self, other):
        '''
        计算两个box的中心点距离
        '''
        return euclidean_distance(self.center, other.center)

    def bound_diagonal_dist(self, other):
        '''
        计算两个box的bound的对角线距离
        '''
        p1 = min(self.x, other.x), min(self.y, other.y)
        p2 = max(self.r, other.r), max(self.b, other.b)
        return euclidean_distance(p1, p2)

    @property
    def location(self):
        return self.x, self.y, self.r, self.b

    @property
    def area(self):
        return self.height * self.width

    @property
    def center(self):
        return [(self.x + self.r) / 2, (self.y + self.b) / 2]



def nms(objs, iou_threshold):
    
    objs = sorted(objs, key=lambda x:x.score, reverse=True)
    removed_flags = [False] * len(objs)
    keep = []
    
    for i in range(len(objs)):
        
        if removed_flags[i]:
            continue
        
        base_box = objs[i]
        keep.append(base_box)
        for j in range(i+1, len(objs)):
            
            if removed_flags[j]:
                continue
            
            other_box = objs[j]
            iou = base_box ^ other_box
            
            if iou > iou_threshold:
                removed_flags[j] = True
    return keep



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

def seed_worker(worker_id):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

_ = one_func_set_all_random_seed(0)


def cal_time(func):
    '''timer'''
    def improved_func(timekeep=False,*args):
        if timekeep:
            start_time = time.time()
            res = func(*args)
            end_time = time.time()
            print('{} takes {}'.format(func.__name__,end_time-start_time))
            return res
        else:
            pass
    return improved_func

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

def one_hot(labels, classes, label_smoothing = 0):
    # refer to 
    n = len(labels) # the number of the samples
    alpha = label_smoothing / classes # 公摊系数
    output = np.ones((n, classes), dtype= np.float32) * alpha

    for row_idx, label in enumerate(labels):
        output[row_idx, label] = 1
    return output

def show_hist(labels, num_classes): # 常用的小工具函数的写法
    label_map = {key: 0 for key in range(num_classes)} # 给每一个类都初始化： 数量为0
    for label in labels:       # 循环labels，遇到label x  就去label x的keyvalue对里+1
        label_map[label] += 1  # 这里相当于是一个一个label item去算
    
    # label_hist 是一个list, list 的值是 label_map key-value 对儿里的 value
    labels_hist = [label_map[key] for key in range(num_classes)]  
    pd.DataFrame(labels_hist, columns=["label"]).plot(kind = "bar") # api 用法的形象记忆 refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
                                                                    # refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html

    evaluate(test_loader, W, b1, U, b2)
    evaluate(test_loader, W, b1, U, b2)
    evaluate(test_loader, W, b1, U, b2)

    evaluate(test_loader, W, b1, U, b2)

def evaluate(test_loader,model, epoch = None, loss = None):
    model.eval()

    correct = 0
    for test_images, test_labels, _ in test_loader:
        probability = softmax(model.inference(test_images))
        predict_labels     = probability.argmax(axis=1).reshape(-1, 1)
        correct       += (predict_labels == test_labels).sum()
    
    acc = correct / len(test_dataset)

    if (epoch is not None) and (loss is not None):
        print(f"{epoch}. train_Loss: {loss:.3f}, test_Accuracy: {acc:.5f}") 
    else:
        print(f"test_Accuracy: {acc:.5f}") 

def sigmoid(x):
    p0 = x < 0
    p1 = ~p0
    x = x.copy()

    # 如果x的类型是整数，那么会造成丢失精度
    x[p0] = np.exp(x[p0]) / (1 + np.exp(x[p0]))
    x[p1] = 1 / (1 + np.exp(-x[p1]))
    return x

def softmax(z):
    z = z.copy()
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    res = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return res

def lr_schedule_cosine(lr_min, lr_max, per_epochs):
    def compute(epoch):
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / per_epochs * np.pi))
    return compute

def estimate_val(predict, gt_labels, classes, loss_func):
    plabel = predict.argmax(1)
    positive = plabel == gt_labels
    total_images = predict.shape[0]
    accuracy = sum(positive) / total_images
    return accuracy, loss_func(predict, one_hot(gt_labels, classes))


    



#endregion
#-------------------------------------------------- 工具函数 ------------------------------------------------------


#-------------------------------------------------- 数据集管理 ------------------------------------------------------
#region
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
    def __init__(self, image_file, label_file, use_conv=True):
        self.num_classes = 10
        self.images = load_images(image_file)
        self.labels = load_labels(label_file)


        if use_conv:    
            # convert 1d to 2d image
            self.images = self.images.reshape(-1, 1, 28, 28)


        self.images = self.images / 255.0
        # self.images = (self.images - np.mean(self.images)) / np.var(self.images)

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
            np.random.shuffle(self.indexes)  # inplace e.g. [0,1,2,3 ....59999] --> [2,1,48,23,...0] 完全复现需要你用的是shuffle 还是 np.shuffle

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
        for index in range(len(batch_data)): # 分别将 img， label， one_hot_label 拼在一起
            batch_data[index] = np.stack(batch_data[index]) # !注意防止1维度的消失, 所以用stack

        return tuple(batch_data)
#endregion
#-------------------------------------------------- 数据集管理 ------------------------------------------------------


#-------------------------------------------------- 计算流程的管理 ------------------------------------------------------
#region

""" 
思考的时候可以参考下面的流程：
    Parameter
    Module
    Linear
    Sigmoid
    SoftmaxCrossEntropyloss
    Network ---> Sequential
    Optimizer
    SGD

 """


class Module:
    """ 
        1.可以称之为算子，那么他应该有forward、backward。为了简化代码，可以用__call__实现forward
        2.需要实现以个params函数，拿出当前module下的所有【递归，如果有子类里面还有子类包含了参数，也要拿出来】参数实例
        3.考虑有些算子，需要感知当前的环境属于训练还是推理，用 train_mode 储存是否为训练状态提供给特定算子用。通过
        train方法和eval方法修改 train_mode 的值

    """
    def __init__(self, name):
        self.name = name
        self.train_mode = False # 为什么需要呢？一个模块的训练模式可能跟测试模式不一样。比如 模拟考搞点意外

    def forward(self, *args):
        # forward输入参数可以是多个
        raise NotImplementedError() # NotImplementedError : https://blog.csdn.net/grey_csdn/article/details/77074707

    def backward(self, grad):
        # 假设算子输出只有一个，所以对应的梯度也应该只有一个
        raise NotImplementedError()


    def __call__(self, *args):  # __call__ ref: http://c.biancheng.net/view/2380.html  # 不定长参数
        return self.forward(*args)

    def train(self):
        self.train_mode = True
        for m in self.modules(): # 什么意思呢？即 你对任意一个module 开启 train mode 其实是对 它的 sub-module 开启train mode
            m.train()

    def eval(self):
        self.train_mode = False
        for m in self.modules():
            m.eval()
    
    def modules(self): # 获取一个模块的所有子模块 这里没有递归
        ms = []
        for attr in self.__dict__: # __dict__  ref: 第一部分 重点理解obj.__dict__即可  https://www.cnblogs.com/starrysky77/p/9102344.html
            m = self.__dict__[attr]
            if isinstance(m, Module):
                ms.append(m)

        return ms

    def params(self): # 获取一个模块所有的参数（如果有的话） 这里有递归
        ps = []
        for attr in self.__dict__:
            p = self.__dict__[attr]     
            if isinstance(p, Parameter): # 先看一下这个p是不是Parameter。kp: 这里用到了递归， 这里是递归的边界  不记得的建议去看一下 python 阶乘 递归
                ps.append(p)            
        
        ms = self.modules()             # 如果不是Parameter的话就直接去找它的子模块（如果有的话）     
        for m in ms:         
            ps.extend(m.params())       # 对所有的子模块依次获取所有参数

        return ps                 

    # -----------------小工具方法（非重点）------------------
    def info(self, n):
        ms = self.modules()  # 拿到所有子模块
        name = self.__class__.__name__ # 拿到当前模块的class 名称
        output = f"{name}\n" 
        for m in ms:         # 下面也是递归
            output += ('  '*(n+1)) + f"{m.info(n+1)}\n" # 顶头缩进，接着以同样的方式去打印它的子模块

        return output[:-1]

    def __repr__(self):
        return self.info(0)
        
class Initializer:
    def __init__(self, name):
        self.name = name
    
    def __call__(self, *args):
        return self.apply(*args)
    
class GaussInitializer(Initializer):
    def __init__(self, mu, sigma):
        super().__init__("GaussInit")
        """ 
         mu: mean
         sigma: the standard deviation. sigma^2 = variance
         """
        self.mu = mu
        self.sigma = sigma

    def apply(self, tensor):
        tensor[...] = np.random.normal(self.mu, self.sigma, tensor.shape)
    
    



class Parameter:
    """ 
    实例化的时候，传入参数值
    封装data、和grad，储存参数值和梯度
        grad.shape = forward(x).shape = batch_size x num_output
        bias.shape = 1 x num_output
     """
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data) # del_W 和 W 的形状要是一样大的

    # 清空参数中存储的梯度
    def zero_grad(self):   # 为什么要清空梯度？ ref: https://blog.csdn.net/weixin_42542536/article/details/104725921
        self.grad[...] = 0 # inplace 操作

#region Conv and Naive Conv

class naive_Conv2d(Module):
    def __init__(self,in_feature, out_feature, kernel_size, padding = 0, stride = 1):
        super().__init__("Conv2d")
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # self.kernel = Parameter(np.zeros((out_feature, in_feature, kernel_size, kernel_size)))# 就是这样定义的
        self.kernel = Parameter(np.random.normal(0, np.sqrt(2 / in_feature), size = (out_feature, in_feature, kernel_size, kernel_size)))

        # self.kernel = Parameter(
        #             np.array([
        #                     [0,0,0],
        #                     [1,1,0],
        #                     [0,0,0] 
        #             ])[None][None])
        self.bias = Parameter(np.zeros((out_feature)))# 每一组kernel 配一个bias

    # @cal_time 
    def forward(self,x):
        # the input :img and kernel
        self.in_shape = x.shape
        ib,ic,ih,iw = self.in_shape
        self.khalf = self.kernel_size//2
        # output
        self.oh = (ih-self.kernel_size + 2*self.padding)//self.stride + 1
        self.ow = (iw-self.kernel_size + 2*self.padding)//self.stride + 1
        self.output = np.zeros((ib,self.out_feature,self.oh,self.ow))
    
        # column
        self.column = np.zeros((ib,self.kernel_size*self.kernel_size*ic ,self.oh*self.ow))
        # k_col
        self.k_col = self.kernel.data.reshape((self.out_feature,-1))

        
        for b in range(ib):
            for channel in range(ic):
                for oy in range(self.oh):# oy ox 指的是输出在输出图像的坐标【跟v1 v2的cy cx不一样】
                    for ox in range(self.ow):
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                            # where the pixel value goes in column
                                column_y = self.kernel_size**2*channel + ky*self.kernel_size + kx
                                column_x = oy*self.ow + ox # ow的格数大小就是kernel横向取了几次
                                # where the pixel value comes from img
                                iy = oy*self.stride+ky - self.padding
                                ix = ox*self.stride+kx - self.padding

                                # 如果iy ix超出边界(可能进入了padding地带)，就不处理
                                if iy >=0 and iy < ih and ix >= 0 and ix < iw:
                                    self.column[b,column_y, column_x] = x[b,channel,iy,ix]
            
            self.output[b] = (self.k_col @ self.column[b]).reshape(-1,self.oh,self.ow) + self.bias.data.reshape((self.out_feature,1,1))       
        return self.output  
        
    def backward(self,G):# G : G_in : dL/d output(this layer)
        ib,ic,ih,iw = self.in_shape # the shape of x  [input of the current layer]
        
        # 1.update part
        # k_col @ column = output
        for b in range(ib):
            # 首先三维的G[b] 肯定是要reshape成2维。因为G[b]：d output(this layer)，所以shape与output[b]是一样的
            # output[b]是[out_feature,oh,ow]
            self.kernel.grad += (G[b].reshape(-1,self.oh*self.ow)@self.column[b].T).reshape(self.kernel.data.shape) # column[b].T shape: (oh*ow,kh*kw*channel)
        
        self.bias.grad += np.sum(G,axis = (0,2,3)) # 因为G的第一个通道是out_feature,对应的就是有多少组kernel
        
        # 2.pass back part
        self.Gout = np.zeros((self.in_shape))

        for b in range(ib):
            # dcolumn我们这里仅仅作为当前图片的dcolumn
            dcolumn = self.k_col.T @ G[b].reshape(self.out_feature,-1) # k_col.T shape: (kw*kh*ic,out_feature)
            # dcolumn 和column shape是一样的
  
            for channel in range(ic):
                for oy in range(self.oh):# oy ox 指的是输出在输出图像的坐标【跟v1 v2的cy cx不一样】
                    for ox in range(self.ow):
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                            # where the pixel value comes from column
                                column_y = self.kernel_size**2*channel + ky*self.kernel_size + kx
                                column_x = oy*self.ow + ox # ow的格数大小就是kernel横向取了几次
                                # where the pixel value goes to img 可参考 notability 笔记 “输入输出坐标的推导”
                                iy = oy*self.stride+ky - self.padding
                                ix = ox*self.stride+kx - self.padding

                                # 如果iy ix超出边界(可能进入了padding地带)，就不处理
                                if iy >=0 and iy < ih and ix >= 0 and ix < iw:
                                    self.Gout[b,channel,iy,ix] += dcolumn[column_y, column_x]
                                    #上面之所以使用+= 是因为在im2col的时候，一个img像素会搬到column的多个地方
                                    #（由于是滑动窗口会重叠），也就是说一个像素会在column不同地方出现，所以回传的时候
                                    #有多个地方贡献梯度
        
        return self.Gout


class Conv2d(Module):

    def __init__(self, in_feature, out_feature, ksize):
        # self.weight = Parameter(
        #     np.zeros((out_feature, in_feature, ksize, ksize)))
        self.weight = Parameter(np.random.normal(0, np.sqrt(2 / in_feature), size = ((out_feature, in_feature, ksize, ksize))))
        self.bias = Parameter(np.zeros((out_feature, 1)))

        self.stride = 1
        self.ksize = ksize

    def forward(self, x):

        self.x = x.copy()
        return self.im2col(x)

    def backward(self, grad):
 
        self.grad = grad.copy()
        return self.col2im(grad)

    def im2col(self, x):
        # refer to 11.39.10.5.jpg
        # 0. info
        self.in_shape = x.shape
        img_num , img_c , img_h, img_w = x.shape


        # 1. W:kcol X:column
        self.kcol = self.weight.data.reshape(self.weight.data.shape[0], -1) # shape[0]-> 多少组卷积
        self.column = np.zeros((img_num, self.kcol.shape[1], (img_h-self.ksize+1) * (img_w-self.ksize +1))) # kcol.shape[1]: kc * kh * kw  e.g. 回忆27 


        # 2. im2col 
        colj = 0
        for icol in range(img_h-self.ksize+1):
            for irow in range(img_w-self.ksize +1):
                self.column[:,:,colj,None] = x[:, :, icol:icol+self.ksize, irow:irow + self.ksize].reshape(img_num,-1,1) # e.g. img_num, 27, 列     这行的None是防止 1 调了
                colj += 1
                
        self.temp_output = self.kcol @ self.column + self.bias.data
        self.temp_out_shape = self.temp_output.shape

        self.output = self.temp_output.reshape(img_num, -1, img_h - self.ksize + 1, img_w - self.ksize + 1)  # 不指定channel
        
        return self.output


    def col2im(self, G):
        G = G.reshape(self.temp_out_shape)  # G: dL / dC dL / d_output , 所以拥有同样的形状  一上来就将G变成 out的形状
        
        # 0. 更新部分
        del_kcol = (G @ self.column.transpose(0,2,1)).sum(axis = 0)  # weight.grad       sum(axis = 0 )指的是将所有的样本贡献的梯度都加起来
        self.weight.grad = del_kcol.reshape(self.weight.data.shape)  # del_W = dL / dW

        # 1. 反传部分
        del_column = self.kcol.T @ G  # del_column 和 column
        del_X = np.zeros(self.in_shape) # 接下去要将del_column 的值正确地填进去del_X里。 Gout: del_X: dL/dx : x :in_shape

        # 2. col2im
        img_num, ic, ih, iw = self.in_shape
        out_fea, in_fea, kh, kw = self.weight.data.shape
        
        del_colj = 0
        for iy in range(0, ih - kh + 1, self.stride):
            for ix in range(0, iw - kw + 1, self.stride):
                select_cols = del_column[: ,:, del_colj,None].reshape(img_num, in_fea, kh, kw) # 选完batch内的所有图片，所有行，del_col_x列
                del_X[:, :, iy:iy + kh, ix:ix + kw] += select_cols
                del_colj += 1
        
        return del_X

#endregion Conv and Naive Conv

class Linear(Module):
    """     
    线性算子，线性层
     职责:
     包含了参数（parameter），包含了运算过程（forward、backward），对于输入的梯度计算，
     和对于参数（parameter）的梯度计算

    """
    def __init__(self, in_feature, out_feature): # 回忆 W  比如: X @ W  = Y   [32x784]@[784x10] = [32x10]
        super().__init__("Linear") # 你如果有继承请一定一定一定要写super
        self.in_feature = in_feature
        self.out_feature = out_feature

        # init method
        # 试一下在这里加入or改成kaiming init
        self.weight = Parameter(np.zeros((in_feature, out_feature)))
        self.bias = Parameter(np.zeros((1, out_feature)))

        # weight init
        initer = GaussInitializer(0, np.sqrt(2 / (in_feature)))
        initer.apply(self.weight.data)

        
    def forward(self, x):
        # 保存x给到backward时使用。 回忆 X @ W = Y    del_W = X^T @ G
        self.x = x

        return x @ self.weight.data + self.bias.data # 回忆一下，加入weight是[784,10] 那么bias形状是多大[10]

    def backward(self, grad):
        """ 
        回忆矩阵乘法：
            X @ W = Y
            del_W = X^T @ G
            del_X = G   @ W^T

         """
        self.weight.grad += self.x.T @ grad  # 回忆一下所说的更新部分  # 为什么要写成+=????
        self.bias.grad   += np.sum(grad, axis = 0, keepdims = True)
        
        return grad @ self.weight.data.T  # 和反传部分

class ReLU(Module):
    def __init__(self, inplace = True):
        super().__init__("ReLU")
        self.inplace = inplace
    
    def forward(self, x):
        self.negative_position = x < 0
        if not self.inplace:
            x = x.copy()
        
        x[self.negative_position] = 0
        return x
    
    def backward(self, G):
        if not self.inplace:
            G = G.copy()               # todo
        
        G[self.negative_position] = 0  # the derivative of relu in <0 is 0
        return G                       # the derivative of relu in >=0 is 1. Thus G * 1

class PReLU(Module):
    def __init__(self, num_feature, inplace=False):
        super().__init__("PReLU")
        self.inplace = inplace
        self.coeff = Parameter(np.zeros((num_feature)))
        
    def forward(self, x):

        if not self.inplace:
            x = x.copy()
            
        for channel in range(x.shape[1]):
            view = x[:, channel]
            negative_position = view < 0
            view[negative_position] *= self.coeff.data[channel]
        return x




class Sigmoid(Module):
    def __init__(self):
        super().__init__("Sigmoid")

    def forward(self, x):
        self.out = self.x = x.copy()
        
        p0 = self.x < 0
        p1 = ~p0
        self.x = self.x.copy()

        self.out[p0] = np.exp(self.x[p0]) / (1 + np.exp(self.x[p0]))
        self.out[p1] = 1 / (1 + np.exp(-self.x[p1]))
        
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)
    

class Swish(Module):
    def __init__(self):
        super().__init__("Swish")

    def forward(self, x):
        self.x_save = x.copy()
        self.sx = sigmoid(x)
        return x * self.sx

    def backward(self, grad):
        return grad * (self.sx + self.x_save * self.sx * (1 - self.sx))


class Dropout(Module):
    def __init__(self, prob_keep=0.5, inplace=True):
        super().__init__("Dropout")
        self.prob_keep = prob_keep
        self.inplace = inplace
        
    def forward(self, x):
        if not self.train_mode:
            return x
        
        self.mask = np.random.binomial(size=x.shape, p=1 - self.prob_keep, n=1)
        if not self.inplace:
            x = x.copy()
            
        x[self.mask] = 0
        x *= 1 / self.prob_keep
        return x
    
    def backward(self, G):
        if not self.inplace:
            G = G.copy()
        G[self.mask] = 0
        G *= 1 / self.prob_keep
        return G



class Maxpool2d(Module):
    def __init__(self, kernel_size = 2, stride = 2):
        super().__init__("Maxpool")
        self.ksize = kernel_size
        self.stride = stride

    def forward(self, x): # not considering padding
        print("Maxpool:",x.shape)
        self.in_shape = x.shape
        ib, ic, ih, iw = x.shape
        # self.oh = (ih - self.ksize)//self.stride + 1 #!    //stride + 1
        # self.ow = (iw - self.ksize)//self.stride + 1
        self.oh = int(np.ceil((ih - self.ksize)/self.stride) + 1) #!    //stride + 1
        self.ow = int(np.ceil((iw - self.ksize)/self.stride) + 1) #!    //stride + 1
        

        output = np.zeros((ib,ic,self.oh,self.ow))

        for oy in range(self.oh):
            for ox in range(self.ow):
                output[:,:, oy, ox] = np.max(x[:, :, oy*self.stride: oy*self.stride + self.ksize,
                                                     ox*self.stride: ox*self.stride + self.ksize ], axis = (2,3)) #! axis = (2,3)

            '''
            No need to worry about the even or odd value of the size.
                let's say we have an x with size (64, 16, 11, 11), given ksz = 2, stride = 2
                11 = 2x5 + 1

                the first 5 indexing will return 
                array([[0., 0.],
                       [0., 0.]])

                But the last one will return 
                array([[0.],
                       [0.]])
            '''
        
        return output
        pass

    def backward(self, G): 
        raise NotImplementedError




class Flatten(Module):
    # flatten仅仅只是将输入一维化，不影响batch大小
    def __init__(self):
        super().__init__("Flatten")

    def forward(self, x):
        self.in_shape = x.shape
        self.out = x.reshape(self.in_shape[0], -1)  # 保留batch大小不变
        return self.out

    def backward(self, G):  # G : dL/dx 所以跟x是一个形状
        return G.reshape(self.in_shape)

class Dropout(Module):
    def __init__(self, pro_keep=0.5, inplace=True):
        super().__init__("Dropout")
        self.pro_keep = pro_keep
        self.inplace = inplace

    def forward(self, x):
        if not self.train_mode:
            return x

        self.mask = np.random.binomial(size=x.shape, p=1 - self.pro_keep, n=1)
        if not inplace:
            x = x.copy()
        x[self.mask] = 0  # 压制住每层false的输入神经元
        x *= 1 / self.pro_keep  # 需要rescale
        return x

    def backward(self, G):
        if not self.train_mode:
            return G

        if not inplace:
            G = G.copy()

        G[self.mask] = 0
        G *= 1 / self.pro_keep
        return G





class SigmoidCrossEntropyLoss(Module):
    def __init__(self, params, weight_decay=1e-5):
        super().__init__("CrossEntropyLoss")
        self.params = params
        self.weight_decay = weight_decay
        
    def sigmoid(self, x):
        #return 1 / (1 + np.exp(-x))
        p0 = x < 0
        p1 = ~p0
        x = x.copy()
        x[p0] = np.exp(x[p0]) / (1 + np.exp(x[p0]))
        x[p1] = 1 / (1 + np.exp(-x[p1]))
        return x
    
    def decay_loss(self):
        loss = 0
        for p in self.params:
            loss += np.sqrt(np.sum(p.data ** 2)) / (2 * p.data.size) * self.weight_decay
        return loss
    
    def decay_backward(self):
        for p in self.params:
            eps = 1e-8
            p.grad += 1 / (2 * np.sqrt(np.sum(p.data ** 2)) + eps) / (2 * p.data.size) * self.weight_decay * 2 * p.data

    def forward(self, x, label_onehot):
        eps = 1e-6
        self.label_onehot = label_onehot
        self.predict = self.sigmoid(x)
        self.predict = np.clip(self.predict, a_max=1-eps, a_min=eps)  # 裁切
        self.batch_size = self.predict.shape[0]
        return -np.sum(label_onehot * np.log(self.predict) + (1 - label_onehot) * 
                        np.log(1 - self.predict)) / self.batch_size + self.decay_loss()
    
    def backward(self):
        self.decay_backward()
        return (self.predict - self.label_onehot) / self.batch_size
    


class SoftmaxCrossEntropyLoss(Module): # 一般的我们会把softmax 跟 loss合起来
    def __init__(self):
        super().__init__("SoftmaxCrossEntropy")
    

    def forward(self, x, label_onehot):
        eps = 1e-6 #! 跟同学对比的时候，你们大家是不是都做了剪裁
        self.label_onehot = label_onehot
        self.predict = softmax(x)
        self.predict = np.clip(self.predict, a_max=1-eps, a_min=eps)  # 裁切
        self.batch_size = self.predict.shape[0]
        return -np.sum(label_onehot * np.log(self.predict)) / self.batch_size #! 搞清楚哪里要除以，哪里不用，不要这里除了，那里又除多一次


    def backward(self,grad = 1):
        return grad * (self.predict - self.label_onehot) / self.batch_size


class Sequential(Module):
    def __init__(self, *items): # 初始化的时候放入一些模块
        super().__init__("Sequential")
        self.items = items

    def modules(self): # 只是返回最浅层即可
        """ 
        覆盖基类的modules方法，直接返回items即可
         """
        return self.items
    
    def forward(self, x):
        # 按照顺序执行items即可
        for m in self.items:
            x = m(x)
        return x

    def backward(self, grad):
        # 按照反向顺序，执行items中模块的backward
        for item in self.items[::-1]: # kp：逆序
            grad = item.backward(grad)
            
        return grad

class Model(Module):
    def __init__(self, use_conv = True): # 可以把Network理解成一个大矩阵。这里num_input就是num_feature 784
        super().__init__("Model")

        # use_conv
        if use_conv:
            self.backbone = Sequential(
                # naive_Conv2d(in_feature = 1, out_feature = 5, kernel_size = 3),
                Conv2d(in_feature = 1, out_feature = 5, kernel_size = 3),
                ReLU(),
                Flatten(),
                Linear(in_feature = 3380, out_feature = 10)
            )

        # BP
        else:
            self.backbone = Sequential(
                Linear(784, 50),
                ReLU(),
                Linear(50, 10)
            )


        

    def forward(self, x):
        return self.backbone(x)

    def backward(self, grad):
        return self.backbone.backward(grad)

    def save(self, f):
        param_list = []
        for p in self.params():
            data = p.data
            grad = p.grad
            param_list.append((data,grad))


        pkl.dump(param_list,open(f"{f}","wb"))
        print(f"the model has been saved.")
        
            
    def load(self, f):
        param_list = pkl.load(open(f"{f}","rb"))
        param_module_list = self.params()

        for idx ,p in enumerate(param_list): # 
            param_module_list[idx].data = p[0] # e.g. (784, 512) 
            param_module_list[idx].grad = p[1] # e.g. (784, 512)

            
        


class Optimizer:
    def __init__(self, name, model, lr):
        self.name = name
        self.model = model
        self.lr = lr
        self.params = model.params()


    def zero_grad(self):
        # 清空所有参数中的梯度
        # 如果需要累计梯度，可以自行控制
        for param in self.params:
            param.zero_grad() # 调用的是class Parameter 的 zero_grad()
    
    def step(self):
        raise NotImplementedError()
    
    def set_lr(self, lr):
        self.lr = lr

class SGD(Optimizer):
    def __init__(self, model, lr=1e-3):
        super().__init__("SGD", model, lr)
    
    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad

class SGDMomentum(Optimizer):
    def __init__(self, model, lr=1e-3, momentum=0.9):
        super().__init__("SGDMomentum", model, lr)
        self.momentum = momentum

        for param in self.params:
            param.v = 0

    # 移动平均
    def step(self):
        for param in self.params:
            param.v = self.momentum * param.v - self.lr * param.grad
            param.data += param.v


class Adam(Optimizer):
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, l2_regularization = 0):
        super().__init__("Adam", model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.l2_regularization = l2_regularization
        self.t = 0
        
        for param in self.params:
            param.m = 0
            param.v = 0
            
    # 指数移动平均
    def step(self):
        eps = 1e-8
        self.t += 1
        for param in self.params:
            g = param.grad
            param.m = self.beta1 * param.m + (1 - self.beta1) * g
            param.v = self.beta2 * param.v + (1 - self.beta2) * g ** 2
            mt_ = param.m / (1 - self.beta1 ** self.t)
            vt_ = param.v / (1 - self.beta2 ** self.t)
            param.data -= self.lr * mt_ / (np.sqrt(vt_) + eps) + self.l2_regularization * param.data

#endregion
#-------------------------------------------------- 计算流程的管理 ------------------------------------------------------



#-------------------------------------------------- MTCNN部分的封装 ------------------------------------------------------

import mtcnn.caffe_pb2 as pb
'''
封装的思考过程：在PNet(归档)这个文件里，我们主要的东西就是

class PNet
    ...
    def load(self)

遵循只要是类似的东西， 我们可以搞出(PNet RNet ONet)，或者抽出来再封装成一个类(def load 和 def fill)。
因此我们决定新封装出以下的类以便简化代码：
1.class Net
2.class CaffeModelLoader

我们希望
class Net  attr: name size (方便net创建一些共有属性）
class CaffeModelLoader  （方便net调用一些共有方法）
我们还希望搞一个最终的net出来，可以直接 net.detect的那种
所以还有一个class MTCNN

'''

class Net(Module):
    def __init__(self, name, cellsize):
        super().__init__(name)
        self.cellsize = cellsize # 感受野大小

class CaffeModelLoader():
    def load(self, file_path, name_and_module_pairs):
        """ 
        file: a string e.g. "mtcnn/det1.caffemodel"    
        思路: 遍历backbone的modules，看它到底是conv 还是prelu
        如果是conv 就fill conv  ----》 fill(conv实例，对应这个名字的层的参数)
        如果是relu 就fill prelu 
         """
        net = pb.NetParameter()

        with open(file_path, "rb") as f: # .caffemodel 存放的是二进制模型参数
            net.ParseFromString(f.read())

        layer_map = {layer.name: layer for layer in net.layer}
        
        for layer_name, layer_module in name_and_module_pairs:
            if isinstance(layer_module, Conv2d):
                self.fill_conv(layer_module, layer_map[layer_name])
            elif isinstance(layer_module, PReLU):
                self.fill_prelu(layer_module, layer_map[layer_name])
            elif isinstance(layer_module, Linear):
                self.fill_linear(layer_module, layer_map[layer_name])  # layer_module是个框架，layer_map[layer_name]是个caffemodel信息体


    def fill_conv(self, layer_ins, layer_info):
        '''
        layer_ins: layer instance from e,g. class Conv2d. No parameters in it.
        layer_info: layer structure containing parameters information which is needed to be deconstructed to access.
        '''
        weight_shape = layer_info.blobs[0].shape.dim
        bias_shape = layer_info.blobs[1].shape.dim[0]
        weights = np.array(layer_info.blobs[0].data)
        bias = np.array(layer_info.blobs[1].data)

        layer_ins.weight.data[...] = weights.reshape(weight_shape)
        layer_ins.bias.data[...] = bias.reshape(bias_shape, 1)

        return layer_ins

    def fill_prelu(self, layer_ins, layer_info):

        coeff_shape = layer_info.blobs[0].shape.dim
        coeff = np.array(layer_info.blobs[0].data)
        
        layer_ins.coeff.data[...] = coeff.reshape(coeff_shape)
        
        return layer_ins
        

    def fill_linear(self, layer_ins, layer_info):
        weight_shape = layer_info.blobs[0].shape.dim
        bias_shape = layer_info.blobs[1].shape.dim
        weights = np.array(layer_info.blobs[0].data)
        bias = np.array(layer_info.blobs[1].data)

        layer_ins.weight.data[...] = weights.reshape(weight_shape).T
        layer_ins.bias.data[...] = bias.reshape(bias_shape)


class PNet(Net, CaffeModelLoader):
    def __init__(self, file):
        super().__init__("PNet", 12)
        self.backbone = Sequential(
            Conv2d(3, 10, 3),
            PReLU(10),
            Maxpool2d(),
            Conv2d(10, 16, 3),
            PReLU(16),
            Conv2d(16, 32, 3),
            PReLU(32)
        )
        self.head_confidence = Conv2d(32, 2, 1)
        self.head_bbox = Conv2d(32, 4, 1)

        self.backbone_names = ["conv1", "PReLU1", "pool1", "conv2", "PReLU2",
                               "conv3", "PReLU3"]
        self.head_names = ["conv4-1", "conv4-2"]
        self.name_and_module_pairs = list(zip(self.backbone_names, self.backbone.modules())) \
                                        + list(zip(self.head_names, [self.head_confidence, self.head_bbox]))

        # 给backbone 和head填权重
        self.load(file, self.name_and_module_pairs)
    
    def forward(self, x):
        print("----------------------------PNET------------------------------")
        x = self.backbone(x)
        return softmax(self.head_confidence(x)), self.head_bbox(x)



class RNet(Net, CaffeModelLoader):
    def __init__(self, file):
        # 希望创建的实例是已经带有权重的了，所以需要导入参数文件
        super().__init__("RNet", 24)
        self.backbone = Sequential(
            Conv2d(3, 28, 3),
            PReLU(28),
            Maxpool2d(kernel_size=3, stride=2),
            Conv2d(28, 48, 3),
            PReLU(48),
            Maxpool2d(kernel_size=3, stride=2),
            Conv2d(48, 64, 2),
            PReLU(64),
            Flatten(),
            Linear(576, 128),  # flatten的576可以手算，也可以i = h(g(f(e(d(c(b(a(x))))))))  abcdefgh分别是上面的层的实例，来算i.shape
            PReLU(128)
        )
        self.head_confidence = Linear(128, 2)
        self.head_bbox = Linear(128, 4)

        self.backbone_names = ["conv1", "prelu1", "pool1", "conv2", "prelu2", "pool2",
                               "conv3", "prelu3", "flatten", "conv4", "prelu4"]
        self.head_names = ["conv5-1", "conv5-2"]
        self.name_and_module_pairs = list(zip(self.backbone_names, self.backbone.modules())) + list(
            zip(self.head_names, [self.head_confidence, self.head_bbox]))

        self.load(file, self.name_and_module_pairs)

    def forward(self, x):
        print("--------------------------------------------RNET--------------------------------------------------")
        x = self.backbone(x)
        return softmax(self.head_confidence(x)), self.head_bbox(x)

class ONet(Net, CaffeModelLoader):
    def __init__(self, file):
        # 希望创建的实例是已经带有权重的了，所以需要导入参数文件
        super().__init__("ONet", 48)
        self.backbone = Sequential(
            Conv2d(3, 32, 3),# conv1
            PReLU(32),
            Maxpool2d(3,2),
            Conv2d(32, 64, 3),# conv2
            PReLU(64),
            Maxpool2d(3,2),
            Conv2d(64, 64, 3),# conv3
            PReLU(64),
            Maxpool2d(2,2),
            Conv2d(64,128,2),
            PReLU(128),
            Flatten(),
            Linear(1152,256),
            PReLU(256)
        )
        self.head_confidence = Linear(256, 2)
        self.head_bbox = Linear(256, 4)
        self.head_landmark = Linear(256,10)

        self.backbone_names = ["conv1", "prelu1", "pool1", "conv2", "prelu2", "pool2",
                               "conv3", "prelu3","pool3","conv4", "prelu4","flatten","conv5",
                               "prelu5"]
        self.head_names = ["conv6-1", "conv6-2","conv6-3"]
        self.name_and_module_pairs = list(zip(self.backbone_names, self.backbone.modules())) + list(
            zip(self.head_names, [self.head_confidence, self.head_bbox,self.head_landmark]))

        self.load(file, self.name_and_module_pairs)

    def forward(self, x):
        print("--------------------------------------------ONET--------------------------------------------------")

        x = self.backbone(x)
        save_var_to_pkl("aout", x)
        return softmax(self.head_confidence(x)), self.head_bbox(x), self.head_landmark(x)


class MTCNN(Module):
    def __init__(self):
        super().__init__("MTCNN")
        self.pnet = PNet("./mtcnn/det1.caffemodel")
        self.rnet = RNet("./mtcnn/det2.caffemodel")
        self.onet = ONet("./mtcnn/det3.caffemodel")

    # @cal_time()
    def pyrdown(self, img,min_face=12,max_face=0,factor=0.709):
        img_h, img_w = img.shape[:2]
        minl = min(img_w, img_h) # 原始图的短边
        scale = 1.0
        scales_select = []

        lower_limit_scale = 12 / max_face if max_face != 0 else 0.01 # 图像大小的下限，其对应的scale是多少 # 注意 12是变换之后，maxface或者minface是变换之前
        upper_limit_scale = 12 / min_face # 图像大小的上限，其对应的scale 是多少
        
        minl *= upper_limit_scale # minl 就从upper limit scale 这个尺度开始了，避免了在过小脸尺度上的搜索
        while minl >= 12.0 and upper_limit_scale * scale >= lower_limit_scale:
            scales_select.append(scale * upper_limit_scale)
            scale *= factor
            minl *= factor
        
        if len(scales_select) == 0:
            return []

        pyrs = []
        for scale in scales_select:
            pyrs.append([cv2.resize(img, (0, 0), fx=scale, fy=scale), scale])
            
        # for p in pyrs:
        #     print(p[0].shape)
        return pyrs
    
    # @cal_time()
    def proposal(self, net, img_pairs, conf_T = 0.6, nms_T = 0.5):
        '''
        net: the net used for proposal
        img_pairs: the input image in the form of [imgs and their scale]
        return bboxes wrt original scale
        '''
        stride = 2
        cellsize = net.cellsize
        bbox_objs = []

        for img, scale in img_pairs:
            img = img.transpose(2,1,0)[None]    # Our parametes come from matlab-based training, which is column-based. Therefore, we need transpose height and width.
            conf,reg = net(img)
            y,x = np.where(conf[0,1]>conf_T)    # the positive output feature map

            # restore the coordinates to the original image to get the location of the bbox detected by PNet
            for oy, ox in zip(y, x):            # for each oy and ox on the output fea_map, they can be restored to the oringinal coordinate system.   
                score = conf[0, 1, oy, ox]      # save the score

                bx = (oy * stride + 1) / scale  # the illustration is refered to "the mapping of the coord" in the folder tutorial image.
                by = (ox * stride + 1) / scale  
                br = (oy * stride + cellsize) / scale
                bb = (ox * stride + cellsize) / scale

                regx = reg[0, 0, oy, ox]
                regy = reg[0, 1, oy, ox]
                regr = reg[0, 2, oy, ox]
                regb = reg[0, 3, oy, ox]

                bw = br - bx + 1
                bh = bb - by + 1

                bx = bx + regx * bw            # the tweak coeffient here is wrt the bw or bh. Note that the bh and bw is the size of the receptive field. 
                by = by + regy * bh
                br = br + regr * bw
                bb = bb + regb * bh
 

                bbox_objs.append(BBox(bx, by, br, bb, score))  # the location of the bboxes are on the scale of original image.
        
        objs = nms(bbox_objs,nms_T)                  # nms on the bboxes

        return objs

    def crop_resize_to(self, src, roi, dst):
        if not roi[0].is_integer():
            int_roi = [int(max(0, float_roi_value)) for float_roi_value in roi]

        else:
            int_roi = roi

        # -------------- ROI cropping ------------------
        x, y, r, b = int_roi
        roi_img = src[y : b,x : r ,:]
        rh, rw, _ = roi_img.shape

        # -------------- Transformation M ------------------
        dst_h, dst_w = dst.shape[:2]
        scale   = min(dst_h/rh, dst_w/rw) # 短边对齐，原图是分母
        translation_w = -rw*0.5*scale + dst_w*0.5
        translation_h = -rh*0.5*scale + dst_h*0.5

        M = np.array([[scale, 0, translation_w],
                        [0, scale, translation_h]])

        # -------------- WarpAffine ------------------
        cv2.warpAffine(roi_img, M, (dst_w, dst_h), dst = dst)
    # @cal_time
    def refine(self,net, objs, src_img, has_landmark = False, conf_T=0.7, nms_T=0.7):
        f'''
        objs: a list consisting of a lot of bbox instances e.g.[bbox1,bbox2,...]

        After PNet, there might be faces detected. If not, return "no face detected". If yes, traverse each bbox instance.
        Then crop and resize the detected bboxes and paste them to a new squre array(e.g. net_cell_size x net_cell_size). Take them as the input of 
        the RNet, run the RNet and get the conf and reg_coeff.

        For refine Net(RNet or ONet), the input images are from the detected bboxes of the upstream net(for RNet, it's PNet; for ONet, it's RNet)
        Then we got input imgs, put them into net and get the confidence and reg_coeff (and probably landmark). Simply speaking, the relationship below holds


                                        crop_resize_to
                                        generate                         predict
                                bbox1                 input_img1                    conf          (15x2x1x1)
                                bbox2  ─────────────► input_img2  ─────►  Net ────► reg_coeff     (15x4x1x1)
                                ...                   ...                           (or landmark) (15x5x2x2)
                                bbox15                input_img15                       |
                                                                                        │filtereed by conf_T
                                                                                        │and NMS
                                                                                        │
                                                                                        ▼
                                                                                    bbox1
                                                                                    bbox3
                                                                                    bbox4
                                                                                    bbox9

        All bboxes are box instance.

        '''


       # 1.generate input imgs from bboxes detected, and do the predction.
        if len(objs) == 0:
            return [],"no face detected after pnet"  # no face detected after pnet

        batch_crp_rsz_imgs =[]         
        for idx, obj in enumerate(objs):                             # obj is a bbox instance                     
            x,y,r,b = obj.location 
            maxl = max(obj.width, obj.height)   
            cx, cy = obj.center
            x = cx - maxl * 0.5
            y = cy - maxl * 0.5
            r = cx + maxl * 0.5
            b = cy + maxl * 0.5           
            dst = np.zeros((net.cellsize, net.cellsize, 3), dtype = np.float32) #! dtype 这里一定要是float，为了承接float型的
            self.crop_resize_to(src_img, (x,y,r,b), dst = dst)  # given an detected box, we get an input image for the net.
            cv2.imwrite(f"./temp_dst/dst_{idx}.jpg",((dst * 128)+127.5).astype(np.uint8)[...,::-1])
            batch_crp_rsz_imgs.append(dst.transpose(2,1,0)[None])  # before inputting to net, imgs require transpose due to column-major principle of matlab

        batch_imgs = np.concatenate(batch_crp_rsz_imgs, axis=0)                 # the cropped & resized images from detected boxes are concanated along the axis of 0(batch axis)
        predicts = net(batch_imgs)                                             
        conf_all,reg_all = predicts[:2]                                         # conf_all, reg_all, landmark(not a must)
        if has_landmark == True:
            landmark = predicts[2]

        # 2.do filtering in terms of confidence and nms.
        keep_objs = []
        for batch_index, obj in enumerate(objs):                     # Each input img is from a bbox instance. The input image is conved down to a pixel which tells us whether the input is a face or not(e.g. think about mayun.jpg 12x12 ---> 1 pixel)
            conf,reg = conf_all[batch_index,1],reg_all[batch_index]  
            if conf > conf_T:                                        # if the pixel value is higher than T.   
                regx,regy,regr,regb = reg[0],reg[1],reg[2],reg[3]    # We tweak the loc of the bbox.
                                                                     # Once again, the tweak coeffient is based on the receptive field. The recpetive field is a square, so we choose a maxline and tweak the xyrb based on the center.
                maxl = max(obj.width, obj.height)   
                cx, cy = obj.center
                x = cx - (0.5 - regx) * maxl  
                y = cy - (0.5 - regy) * maxl
                r = cx + (0.5 + regr) * maxl
                b = cy + (0.5 + regb) * maxl
                
                new_obj = BBox(x, y, r, b, conf)                      # and also we create a new bbox with the tweaked location.

                if has_landmark:
                    lmks_one_face = landmark[batch_index]            # for each image(specified by the batch_index), we get 
                    new_obj.landmarks = []                           # add a new feature(landmarks) to class bbox.
                    for lmk_x,lmk_y in zip(lmks_one_face[:5],lmks_one_face[5:]): # the first 5 are x, then they are y. Pair them up.
                        lmk_x = cx - (0.5 - lmk_x) * maxl                             # lmk_x and lmm_y on the right of the equation are just coeffients.
                        lmk_y = cy - (0.5 - lmk_y) * maxl
                        new_obj.landmarks.append((lmk_x,lmk_y))                   

                keep_objs.append(new_obj)                            # The bboxes satisifing conf_T are tweaked for a better xyrb.
        
        return nms(keep_objs,nms_T)


    def detect(self,image,min_face=30,max_face=0):
        '''
        detect要考虑输入图像，并组建图像金字塔，然后再依次通过三个网络
        此处的min_face与pyrdown的min_face=12 略有不同，这里选的30是根据
        实际应用的情况，12是因为网络自身结构的原因
        '''
        self.input_img = image[...,::-1]# bgr2rgb
        self.input_img = ((self.input_img-127.5)/128.0).astype(np.float32)  # Note that the normalization and float32
        self.pyrs = self.pyrdown(self.input_img,min_face,max_face)
        self.objs = self.proposal(self.pnet, self.pyrs,conf_T=0.6,nms_T=0.6)
        self.robjs = self.refine(self.rnet, self.objs,self.input_img,conf_T=0.7,nms_T=0.6)
        self.oobjs = self.refine(self.onet,self.robjs,self.input_img, has_landmark=True,conf_T=0.7,nms_T = 0.5)
        return self.oobjs





#-------------------------------------------------- MTCNN部分的封装 ------------------------------------------------------



import time

def draw_landmarks(bbox, src_img):
    """
    bbox: a bbox instance that comes with x, y, r, b, score and landmarks
    src_img: src imag on which the landmarks are drawn.
    """
    for lmk_x, lmk_y in bbox.landmarks:
        cv2.circle(src_img, (int(lmk_x), int(lmk_y)), radius = 2, color = (0,255,0), thickness = 2)


def inference():
    mtcnn = MTCNN()
    img_name = "rq.jpg"
    show_conf = True

    image = cv2.imread(f"{img_name}")
    final_objs = mtcnn.detect(image, 30, 50)

    # draw bbox for the detected objs.
    for idx ,obj in enumerate(final_objs):
        bx,by,br,bb = np.round(obj.location).astype(np.int32)
        cv2.rectangle(image,(bx,by),(br,bb),(0,255,0),2)
        draw_landmarks(obj, image) # the loc of the landmarks

        if show_conf:
            cv2.putText(image, "{:.2f}".format(obj.score), (bx, by), fontFace = 0, fontScale = 0.5, color = (255,0,0), thickness= 2 )
        
    cv2.imwrite(f"my_{img_name}",image)



if __name__  == "__main__":
    inference()
    
#     #-------------------- debug Conv2d -----------------------------
#     x = load_var_from_pkl("ax")
#     print(x.shape)
#     x = x[0:1,0:1]
#     print(x.shape)
#     conv = Conv2d(1, 1, kernel_size= 3)
#     out = conv(x)
#     print(out)