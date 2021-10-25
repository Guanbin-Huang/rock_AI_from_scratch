import numpy as np
import struct
import matplotlib.pyplot as plt



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


def load_labels(file):   # 加载数据
    with open(file, "rb") as f :
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)

def load_images(file):   # 加载数据
    with open(file,"rb") as f :
        data = f.read()
    magic_number , num_items, rows, cols = struct.unpack(">iiii",data[:16])
    return  np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items,-1)

def make_onehot(array_x,class_nums = 10):
    datas_len = len(array_x)

    result = np.zeros((datas_len,class_nums))

    for index,n  in enumerate(array_x):
        # d = array_x[index]
        result[index][n]   =   1

    return result


def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex,axis = 1, keepdims = True)

class Dataset():
    def __init__(self, img_data, img_label, batch_size,shuffle):
        self.img_data = img_data
        self.img_label = img_label
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return Dataloader(self)

    def __len__(self):
        return len(self.img_data)

class Dataloader():
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0
        self.indexs = np.arange(len(self.dataset))

        if self.dataset.shuffle:
            np.random.shuffle(self.indexs)

    def __next__(self):
        if self.cursor>=len(self.dataset):
            raise StopIteration

        indexs = self.indexs[self.cursor:self.cursor+self.dataset.batch_size]

        x = self.dataset.img_data[indexs]
        y = self.dataset.img_label[indexs]

        self.cursor += self.dataset.batch_size

        return x, y



if __name__ == "__main__":

    # z-score

    train_data = load_images("dataset/train-images-idx3-ubyte")
    train_data = train_data/255
    train_label = make_onehot(load_labels("dataset/train-labels-idx1-ubyte"))


    test_data = (load_images("dataset/t10k-images-idx3-ubyte"))/255
    test_label = load_labels("dataset/t10k-labels-idx1-ubyte")

    train_num, feature_num = train_data.shape
    class_num = 10

    weight = np.random.normal(0,1,size = (feature_num,class_num))
    b = 0 
    lr = 0.1
    epoch = 1000
    batch_size = 256
    shuffle = True
    dataset = Dataset(train_data,train_label,batch_size,shuffle)
    # lr_d = {0:1, 300:0.1, 500:0.01, 700:0.001}

    for e in range(epoch):
        # if e in lr_d:
        #     lr = lr_d[e]
        for x, y in dataset:
            p = x @ weight + b

            predict = softmax(p)

            loss = - np.sum(y * np.log(predict)) / batch_size         # 交叉熵loss

            G = predict - y

            delta_weight = (x.T @ G)/batch_size
            delta_b = np.sum(G)/batch_size

            weight -= lr * delta_weight
            b -= lr * delta_b

        if True:
            t_p = test_data @ weight + b
            p_label = np.argmax(t_p, axis= 1)
            acc = np.sum(p_label == test_label) / len(test_label)
            print(f"loss:{loss}||||acc:{acc*100:.3f}%|||lr:{lr:.3f}")
