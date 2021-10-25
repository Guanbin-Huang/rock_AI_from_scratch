import numpy as np
import struct
import random
# import matplotlib.pyplot as plt
# import pandas as pd
import math



def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()

    magic_number, num_samples = struct.unpack(">ii", data[:8])
    if magic_number != 2049:  # 0x00000801
        return None

    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels

def load_images(file):
    with open(file, "rb") as f:
        data = f.read()

    magic_number, num_samples, image_width, image_height = struct.unpack(">iiii", data[:16])
    if magic_number != 2051:  # 0x00000803
        print(f"magic number mismatch {magic_number} != 2051")
        return None

    image_data = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_samples, -1) # np.frombuffer(读二进制)
    return image_data

def one_hot(labels, classes, label_smoothing=0):
    n = len(labels)
    eoff = label_smoothing / classes
    output = np.ones((n, classes), dtype=np.float32) * eoff
    for row, label in enumerate(labels):
        output[row, label] = 1 - label_smoothing + eoff
    return output

class Dataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    # 获取他的一个item，  dataset = Dataset(),   dataset[index]
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    # 获取数据集的长度，个数
    def __len__(self):
        return len(self.images)

class DataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.cursor = 0
        self.indexs = list(range(self.dataloader.count_data))  # 0, ... 60000
        if self.dataloader.shuffle:
            # 打乱一下
            random.shuffle(self.indexs)

    def __next__(self):
        if self.cursor >= self.dataloader.count_data:
            raise StopIteration()

        batch_data = []
        remain = min(self.dataloader.batch_size, self.dataloader.count_data - self.cursor)  # 256, 128
        for n in range(remain):
            index = self.indexs[self.cursor]
            data = self.dataloader.dataset[index]

            # 如果batch没有初始化，则初始化n个list成员
            if len(batch_data) == 0:
                batch_data = [[] for i in range(len(data))]

            # 直接append进去
            for index, item in enumerate(data):
                batch_data[index].append(item)
            self.cursor += 1

        # 通过np.vstack一次性实现合并，而非每次一直在合并
        for index in range(len(batch_data)):
            batch_data[index] = np.vstack(batch_data[index])
        return batch_data

class DataLoader:

    # shuffle 打乱
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.shuffle = shuffle
        self.count_data = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        return DataLoaderIterator(self)


def estimate(plabel, gt_labels, classes):
    plabel = plabel.copy()
    gt_labels = gt_labels.copy()
    match_mask = plabel == classes
    mismatch_mask = plabel != classes
    plabel[match_mask] = 1
    plabel[mismatch_mask] = 0

    gt_mask = gt_labels == classes
    gt_mismatch_mask = gt_labels != classes
    gt_labels[gt_mask] = 1
    gt_labels[gt_mismatch_mask] = 0

    TP = sum(plabel & gt_labels)
    FP = sum(plabel & (1 - gt_labels))
    FN = sum((1 - plabel) & gt_labels)
    TN = sum((1 - plabel) & (1 - gt_labels))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, accuracy, F1

def estimate_val(predict, gt_labels, classes, loss_func):
    plabel = predict.argmax(1)
    positive = plabel == gt_labels
    total_images = predict.shape[0]
    accuracy = sum(positive) / total_images
    return accuracy, loss_func(predict, one_hot(gt_labels, classes))


def lr_cosine_schedule(lr_min, lr_max, Ti):
    '''
    :param Ti: Ti epochs are performed before a new restart.
    :param Tcur: How many epochs have been performed since the last restart.
    :return: a function to compute a value within a period.
    '''
    def compute(Tcur):
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(Tcur / Ti * np.pi))
    return compute



def sigmoid(x):
    p0= x<0
    p1 = ~p0 # 补集
    x = x.copy()
    x[p0] = np.exp(x[p0])/(np.exp(x[p0])+1)
    x[p1] = 1/(1+np.exp(-x[p1]))
    return x

def softmax(x):
    x = x.copy()
    x_max = np.max(x,axis = 1)
    exp_x = np.exp(x-x_max)
    return exp_x /np.sum(exp_x,axis = 1,keepdims = True)


def cross_entropy(predict, gt):
    eps = 1e-4
    predict = np.clip(predict, a_max=1-eps, a_min=eps)  # 裁切
    batch_size = predict.shape[0]
    return -np.sum(gt * np.log(predict) + (1 - gt) * np.log(1 - predict)) / batch_size # loss for one batch 不能用mean，因为mean 会对整个矩阵求mean，分母（除以的）是所有像素数量，但是我们谈论loss 的时候是平均每个样本的loss，所以只能除以batch_size


class Module:
    def __init__(self, name):
        self.name = name
        self.train_mode = False

    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        self.train_mode = True
        for m in self.modules():
            m.train()

    def eval(self):
        self.train_mode = False
        for m in self.modules():
            m.eval()

    def modules(self):                 # 用来拿到一个module下的submodule
        ms = []
        for attr in self.__dict__:
            m = self.__dict__[attr]
            if isinstance(m, Module):
                ms.append(m)
        return ms

    def params(self):
        ps = []                          # ps: parameters
        for attr in self.__dict__:
            p = self.__dict__[attr]      # p: 可能是paramter的obj
            if isinstance(p, Parameter):
                ps.append(p)

        ms = self.modules()              # 如果self是model，那么
        for m in ms:
            ps.extend(m.params())
        return ps

    def info(self, n):
        ms = self.modules()
        output = f"{self.name}\n"
        for m in ms:
            output += (' ' * (n + 1)) + f"{m.info(n + 1)}\n"
        return output[:-1]

    def __repr__(self):
        return self.info(0)

class ModuleList(Module):
    def __init__(self, *args):
        super().__init__("ModuleList")
        self.ms = list(args)

    def __repr__(self):
        return f'self.ms'

    def modules(self):
        return self.ms

    def forward(self, x):
        for m in self.ms:
            x = m(x)
        return x

    def backward(self, G):
        for i in range(len(self.ms) - 1, -1, -1):
            G = self.ms[i].backward(G)
        return G

class Model(Module):
    def __init__(self, num_feature, num_hidden, num_classes):
        super().__init__("Model")
        self.backbone = ModuleList(
            Linear(num_feature, num_hidden),
            ReLU(),
            # Sigmoid(),
            Linear(num_hidden, num_classes)
        )


    def forward(self, x):
        return self.backbone(x)

    def backward(self, G):
        return self.backbone.backward(G)

class Initializer:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args):
        return self.apply(*args)

class GaussInitializer(Initializer):
    # where :math:`\mu` is the mean and :math:`\sigma` the standard
    # deviation. The square of the standard deviation, :math:`\sigma^2`,
    # is called the variance.
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def apply(self, value):
        np.random.seed(3)
        value[...] = np.random.normal(self.mu, self.sigma, value.shape)

class Parameter:
    def __init__(self, value):
        self.value = value
        self.delta = np.zeros(value.shape)

    def zero_grad(self):
        self.delta[...] = 0

class Linear(Module):
    def __init__(self, input_feature, output_feature):
        super().__init__("Linear")
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.weights = Parameter(np.zeros((input_feature, output_feature))) # 参数们都是Parameter类
        self.bias = Parameter(np.zeros((1, output_feature)))

        # 权重初始化
        # initer = GaussInitializer(0, np.sqrt(2 / input_feature))# kaiming初始化
        initer = GaussInitializer(0, 1)
        initer.apply(self.weights.value)

    def forward(self, x):
        self.x_save = x.copy()
        
        return x @ self.weights.value + self.bias.value


    # AB = C  G
    # dB = A.T @ G
    # dA = G @ B.T
    def backward(self, G):
        self.weights.delta += self.x_save.T @ G
        # +=是因为考虑了多个batch后再更新；这里不用/batch_size 是因为回传的第一个G
        # 也就是loss 的G 已经除以了batchsize 了。
        self.bias.delta += np.sum(G, 0)  # 值复制
        return G @ self.weights.value.T

class ReLU(Module):
    def __init__(self, inplace=True):
        super().__init__("ReLU")
        self.inplace = inplace

    def forward(self, x):
        self.x_negative = x < 0
        if not self.inplace:
            x = x.copy()

        x[self.x_negative] = 0
        return x

    def backward(self, G):
        if not self.inplace:
            G = G.copy()

        G[self.x_negative] = 0  # 这里不是G[G<0] 而是 G[x<0] = 0 因为 G * d_Relu(out)/d_out   #   
        return G


class Sigmoid(Module):
    def __init__(self, inplace=True):
        super().__init__("Sigmoid")
        self.inplace = inplace

    def forward(self, x):
        self.x_save = x.copy()
        x = sigmoid(x)
        return x

    def backward(self, G):
        return G * sigmoid(self.x_save) * (1 - sigmoid(self.x_save))


class Dropout(Module):
    def __init__(self,pro_keep = 0.5, inplace = True):
        super().__init__("Dropout")
        self.pro_keep = pro_keep
        self.inplace = inplace

    def forward(self,x):
        if not self.train_mode:
            return x

        self.mask = np.random.binomial(size = x.shape, p = 1-self.pro_keep, n =1)
        if not inplace:
            x = x.copy()
        x[self.mask] = 0# 压制住每层false的输入神经元
        x *= 1/self.pro_keep # 需要rescale
        return x

    def backward(self,G):
        if not self.train_mode:
            return G

        if not inplace:
            G = G.copy()

        G[self.mask] = 0
        G *= 1/self.pro_keep
        return G


class SigmoidCrossEntropy(Module):
    def __init__(self):
        super().__init__("CrossEntropyLoss")

    def forward(self,x,one_hot_labels):
        self.labels = one_hot_labels
        self.predict = sigmoid(x)
        self.batch_size = self.predict.shape[0]
        loss = cross_entropy(self.predict,self.labels) # loss for one batch   在cross_entropy里面已经除了batchsize了
        return loss

    def backward(self):
        return (self.predict - self.labels)/self.batch_size

class SoftmaxCrossEntropy(Module):
    def __init__(self):
        super().__init__("SoftmaxCrossEntropy")

    def forward(self,x,one_hot_labels):
        self.predict = softmax(x)
        self.labels = one_hot_labels
        self.batch_size = self.predict.shape[0]
        loss = cross_entropy(self.predict,self.labels)/self.batch_size # loss for one batch
        return loss


class Optimizer:
    def __init__(self, name, model, lr):
        self.name = name
        self.model = model
        self.lr = lr
        self.params = model.params() # 能进行model.params()这个操作是因为， module 基类有 def params():

    def zero_grad(self): # 最开始先有Parameter类的zero_grad. 由于Optimizer里都存有parameters，所以让optimizer zero_grad，其实就是让优化器里的参数zero_grad
        for param in self.params:
            param.zero_grad()

    def set_lr(self, lr):
        self.lr = lr

class SGD(Optimizer):
    def __init__(self, model, lr=1e-3):
        super().__init__("SGD", model, lr)

    def step(self):
        for param in self.params:
            param.value -= self.lr * param.delta

class Adam(Optimizer):# l2 和adam不要一起用 https://zhuanlan.zhihu.com/p/63982470
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999):
        super().__init__("Adam", model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

        for param in self.params:
            param.m = 0 # w和b参数多了m v两个属性
            param.v = 0

    def step(self):
        eps = 1e-8
        self.t += 1
        for param in self.params:
            g = param.delta
            param.m = self.beta1 * param.m +(1-self.beta1)*g
            param.v = self.beta2* param.v  + (1-self.beta2)*g**2

            param.m_ = param.m/(1-self.beta1**self.t)
            param.v_ = param.v/(1-self.beta2**self.t)

            param.value -= self.lr*param.m_/(np.sqrt(param.v_)+ eps)



def estimate_val(predict, gt_labels, classes, loss_func):
    plabel = predict.argmax(1)
    positive = plabel == gt_labels
    total_images = predict.shape[0]
    accuracy = sum(positive) / total_images
    return accuracy, loss_func(predict, one_hot(gt_labels, classes))

def lr_schedule_cosine(lr_min, lr_max, per_epochs):
    def compute(epoch):
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / per_epochs * np.pi))

    return compute

def min_max_normalize(images):
    return (images - np.min(images))/(np.max(images) - np.min(images))

def standardize(images):
    return (images - np.mean(images))/(np.var(images))

def draw_histogram(img_1d_arr):
    # hist,bins= np.histogram(img,bins=20,range=(-0.5,0.5))
    plt.hist(img_1d_arr, bins=20,range=(-1,1))
    plt.title("histogram")
    plt.show()

def load_cifar10(directory,num_batches_to_be_loaded=5):
    import numpy as np
    import os
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    #1. load the train_data
    train_img_list = []
    train_label_list = []
    for i in range(1,num_batches_to_be_loaded + 1):
        file_name = os.path.join(directory,f"data_batch_{i}")
        dict = unpickle(file_name)
        _,labels,datas,_ = dict[b'batch_label'],dict[b'labels'],dict[b'data'],dict[b'filenames']

        train_img_list.append(datas)
        train_label_list.append(labels)

    train_imgs = np.concatenate(train_img_list,axis = 0)
    train_labels = np.concatenate(train_label_list)

    #2. load the test data
    file_name = os.path.join(directory, f"test_batch")
    dict = unpickle(file_name)
    _,val_labels,val_imgs,_ = dict[b'batch_label'],dict[b'labels'],dict[b'data'],dict[b'filenames']
    val_labels = np.array(val_labels)

    return train_imgs,train_labels,val_imgs,val_labels


def set_seed(seed):
    import random
    import os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)