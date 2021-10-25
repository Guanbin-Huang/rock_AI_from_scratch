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

class MyDataset():
    def __init__(self,img_data,img_label,batch_size, shuffle):
        self.img_data = img_data
        self.img_label = img_label
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return DataLoader(self)

    def __len__(self):
        return len(self.img_data)

class DataLoader():
    def __init__(self,dataset):
        self.dataset = dataset
        self.cursor = 0

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        indexs = np.arange(len(self.dataset))

        if self.dataset.shuffle :
            np.random.shuffle(indexs)

        index = indexs[self.cursor:self.cursor + self.dataset.batch_size]

        x = self.dataset.img_data[index]
        y = self.dataset.img_label[index]

        self.cursor +=   self.dataset.batch_size

        return x ,y


if __name__ == "__main__":

    # z-score

    train_data = load_images("dataset/train-images-idx3-ubyte")
    train_data = train_data/255
    train_label = make_onehot(load_labels("dataset/train-labels-idx1-ubyte"))

    test_data = (load_images("dataset/t10k-images-idx3-ubyte"))/255
    test_label = load_labels("dataset/t10k-labels-idx1-ubyte")

    train_num, feature_num = train_data.shape
    class_num = 10
    last_acc = 0
    hidden_num = 256


    weight1 = np.random.normal(0,1,size = (feature_num,hidden_num))
    weight2= np.random.normal(0,1,size = (hidden_num,class_num))

    bias1 = np.zeros((1,weight1.shape[-1]))
    bias2 = np.zeros((1,weight2.shape[-1]))


    lr = 0.1
    epoch = 1000
    batch_size = 100
    shuffle = True

    k = 0.01
    

    dataset = MyDataset(train_data,train_label,batch_size,shuffle)
    for e in range(epoch):
        for x,y in dataset:
            hidden = x @ weight1 + bias1

            # 加一层激活函数
            hidden_k = k * hidden   # 换成 像样的激活函数 : sigmoid, tanh, softmax??  relu, prelu

            p = hidden_k @ weight2  + bias2

            predict = softmax(p)
            loss = -np.sum(y * np.log(predict)) /batch_size

            G2 = (predict - y)/batch_size

            delta_w2 = hidden.T @ G2
            G1 = delta_hidden = (G2 @ weight2.T) /batch_size

            delta_w1 = x.T @ G1

            delta_b1 = np.sum(G1,axis = 0,keepdims = True) / batch_size
            delta_b2 = np.sum(G2,axis = 0, keepdims = True) / batch_size


            weight1 -= lr * delta_w1
            weight2 -= lr * delta_w2

            bias1 -= lr * delta_b1
            bias2 -= lr * delta_b2


        if e % 1 == 0:
            t_h = test_data @ weight1 + bias1
            t_P = (t_h * k) @ weight2 + bias2

            # t_predict = softmax(tp)           #  10000 * 10
            p_label = np.argmax(t_P,axis = 1)
            acc = np.sum(p_label == test_label) / 100

            print(f"acc : {acc:.3f} %, lr : {lr},  loss : {loss:.3f}")

                # if acc < last_acc :   # 根据准确率(loss), 调整 lr , 自适应调整学习率
                #     lr /= 2
                # else:
                #     last_acc = acc

