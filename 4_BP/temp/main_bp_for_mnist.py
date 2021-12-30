from utils import *
import numpy as np
import struct
import random
import matplotlib.pyplot as plt
import pandas as pd
import math


train_labels = load_labels('./dataset/train-labels-idx1-ubyte')  # 60000,
train_images = load_images('./dataset/train-images-idx3-ubyte')  # 60000, 784
train_images = train_images / 255 # 归一化特征尺度
train_images = (train_images - np.mean(train_images)) / np.var(train_images) # 让输入服从高斯分布


val_labels = load_labels('./dataset/t10k-labels-idx1-ubyte')  # 10000,
val_images = load_images('./dataset/t10k-images-idx3-ubyte')  # 10000, 784
val_images = val_images / 255
val_images = (val_images - np.mean(val_images)) / np.var(val_images) # 除以方差还是标准差实验一下



# np.random.seed(3)
classes = 10  # 定义10个类别
batch_size = 64  # 定义每个批次的大小
epochs = 20  # 退出策略，也就是最大把所有数据看10次
lr = 1e-2
numdata, data_dims = train_images.shape  # 60000, 784

# 定义dataloader和dataset，用于数据抓取
train_data = DataLoader(Dataset(train_images, one_hot(train_labels, classes)), batch_size, shuffle=True)
model = Model(data_dims, 1024, classes)

loss_func = SigmoidCrossEntropy()

optim = SGD(model, lr)
iters = 0  # 定义迭代次数，因为我们需要展示loss曲线，那么x将会是iters

lr_schedule = {
    5: 1e-3,
    15: 1e-4,
    18: 1e-5
}

# 开始进行epoch循环，总数是epochs次
print("training starts")
for epoch in range(epochs):
    if epoch in lr_schedule:
#         LR = lr_schedule[epoch]
        optim.set_lr(lr)

    model.train()
    # 对一个批次内的数据进行迭代，每一次迭代都是一个batch（即256）
    for index, (images, labels) in enumerate(train_data):
        x = model(images)

        # 计算loss值
        loss = loss_func(x, labels)
        optim.zero_grad()
        G = loss_func.backward()
        model.backward(G)
        optim.step()   # 应用梯度，更新参数
        iters += 1

    print(f"Iter {iters}, {epoch} / {epochs}, Loss {loss:.3f}, LR {lr:g}")

    model.eval()
    val_accuracy, val_loss = estimate_val(model(val_images), val_labels, classes, loss_func)
    print(f"{epoch} Val set, Accuracy: {val_accuracy:.6f}, Loss: {val_loss:.3f}")
