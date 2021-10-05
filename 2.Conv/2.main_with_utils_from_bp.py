from utils_from_bp import *
from tqdm import tqdm


if __name__ == "__main__":

    # ------------------------------- HYPER PARAMETER SETTING ---------------------------------
    classes = 10  # 定义10个类别
    batch_size = 256  # 定义每个批次的大小
    num_epochs = 10  # 退出策略，也就是最大把所有数据看10次
    lr = 0.1
    use_conv = False 
    lr_schedule = {
        5: 1e-3,
        15: 1e-4,
        18: 1e-5
    }


    # ------------------------------- DATA LOADING ---------------------------------

    print("dataset is ready")
    # 定义dataloader和dataset，用于数据抓取
    train_dataset = MNIST_Dataset("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte", use_conv)
    train_loader  = DataLoader(train_dataset, batch_size, True)
    
    # 选择一次性全部验证，快一点
    val_images = load_images("dataset/t10k-images-idx3-ubyte") # 10000, 784
    val_labels = load_labels("dataset/t10k-labels-idx1-ubyte") # 10000
    val_images = val_images/255
    # val_images = (val_images - 0.13066047) / 0.09493
    if use_conv:
        val_images = val_images.reshape(-1,1,28,28)





    # ------------------------------ OPTION: CNN or BP -----------------------------------
    if len(train_dataset.images.shape) >=4:
        train_numdata ,channels, img_h, img_w = train_dataset.images.shape
    else:
        train_numdata, img_size = train_dataset.images.shape

    # iter_per_epoch = numdata//batch_size if numdata % batch_size == 0 else (numdata // batch_size + 1)
    iter_per_epoch = len(train_loader)



    # ------------------------------- MODEL AND OPTIMIZER ---------------------------------

    model = Model(use_conv = use_conv)
    optim = SGD(model, lr)
    loss_func = SoftmaxCrossEntropyLoss()
    iters = 0  # 定义迭代次数，因为我们需要展示loss曲线，那么x将会是iters



    # ---------------------------------------------training and validation------------------------------
    print(f"start training: \ninitial lr:{lr}  batchsize {batch_size}  epochs: {num_epochs} use_conv: {use_conv}")

    # 开始进行epoch循环，总数是epochs次
    for epoch in range(num_epochs):
    #     if epoch in lr_schedule:
    # #         lr = lr_schedule[epoch]
    #         optim.set_lr(lr)
        if epoch == 8:
            lr *= 0.1

        model.train()
        cur_iter = 0
        # 对一个批次内的数据进行迭代，每一次迭代都是一个batch（即256）ba
        for index, (images, _, onehot_labels) in enumerate(train_loader):
            cur_iter += 1

            x = model(images)

            # 计算loss值
            loss = loss_func(x, onehot_labels)
            
            # print(loss)
            # if index == 5:
            #     exit()
            optim.zero_grad()
            G = loss_func.backward()
            model.backward(G)
            optim.step()   # 应用梯度，更新参数
            iters += 1

        print(f"Iter {iters}, {epoch} / {num_epochs}, Loss {loss:.3f}, LR {lr:g}")

        model.eval()
        val_accuracy, val_loss = estimate_val(model(val_images), val_labels, classes, loss_func)
        print(f"Val set, Accuracy: {val_accuracy:.6f}, Loss: {val_loss:.3f}")



""" 
接着提升的方向和尝试：
    SGD momentum   adam
    每一轮都打乱index
    ref: https://github.com/shouxieai/bp-cpp/blob/main/src/main.cpp
    etc


 """


""" 
下面全都用的是SGD

dataset is ready
start training: 
initial lr:0.01  batchsize 32  epochs: 10 use_conv: True
Iter 1875, 0 / 10, Loss 0.217, LR 0.01
Val set, Accuracy: 0.941600, Loss: 0.210
Iter 3750, 1 / 10, Loss 0.116, LR 0.01
Val set, Accuracy: 0.950000, Loss: 0.167
Iter 5625, 2 / 10, Loss 0.159, LR 0.01
Val set, Accuracy: 0.960500, Loss: 0.139
Iter 7500, 3 / 10, Loss 0.127, LR 0.01
Val set, Accuracy: 0.966900, Loss: 0.112
Iter 9375, 4 / 10, Loss 0.047, LR 0.01
Val set, Accuracy: 0.967800, Loss: 0.103
Iter 11250, 5 / 10, Loss 0.066, LR 0.01
Val set, Accuracy: 0.969600, Loss: 0.097
Iter 13125, 6 / 10, Loss 0.157, LR 0.01
Val set, Accuracy: 0.971700, Loss: 0.093
Iter 15000, 7 / 10, Loss 0.047, LR 0.001
Val set, Accuracy: 0.972700, Loss: 0.092
Iter 16875, 8 / 10, Loss 0.172, LR 0.001
Val set, Accuracy: 0.972800, Loss: 0.087
Iter 18750, 9 / 10, Loss 0.035, LR 0.001
Val set, Accuracy: 0.975300, Loss: 0.080




start training: 
initial lr:0.01  batchsize 32  epochs: 10 use_conv: False
Iter 1875, 0 / 10, Loss 0.376, LR 0.01
Val set, Accuracy: 0.901000, Loss: 0.365
Iter 3750, 1 / 10, Loss 0.141, LR 0.01
Val set, Accuracy: 0.913500, Loss: 0.304
Iter 5625, 2 / 10, Loss 0.318, LR 0.01
Val set, Accuracy: 0.922700, Loss: 0.276
Iter 7500, 3 / 10, Loss 0.112, LR 0.01
Val set, Accuracy: 0.927500, Loss: 0.261
Iter 9375, 4 / 10, Loss 0.315, LR 0.01
Val set, Accuracy: 0.930600, Loss: 0.243
Iter 11250, 5 / 10, Loss 0.058, LR 0.01
Val set, Accuracy: 0.934100, Loss: 0.232
Iter 13125, 6 / 10, Loss 0.133, LR 0.01
Val set, Accuracy: 0.939000, Loss: 0.217
Iter 15000, 7 / 10, Loss 0.099, LR 0.001
Val set, Accuracy: 0.940500, Loss: 0.205
Iter 16875, 8 / 10, Loss 0.401, LR 0.001
Val set, Accuracy: 0.943200, Loss: 0.197
Iter 18750, 9 / 10, Loss 0.590, LR 0.001
Val set, Accuracy: 0.945000, Loss: 0.188
 
 




dataset is ready
start training: 
initial lr:1  batchsize 256  epochs: 10 use_conv: False
Iter 234, 0 / 10, Loss 0.246, LR 1
Val set, Accuracy: 0.923000, Loss: 0.246
Iter 468, 1 / 10, Loss 0.194, LR 1
Val set, Accuracy: 0.942100, Loss: 0.188
Iter 702, 2 / 10, Loss 0.127, LR 1
Val set, Accuracy: 0.952400, Loss: 0.157
Iter 936, 3 / 10, Loss 0.274, LR 0.1
Val set, Accuracy: 0.951800, Loss: 0.158
Iter 1170, 4 / 10, Loss 0.126, LR 0.1
Val set, Accuracy: 0.954600, Loss: 0.149
Iter 1404, 5 / 10, Loss 0.232, LR 0.1
Val set, Accuracy: 0.952200, Loss: 0.156
Iter 1638, 6 / 10, Loss 0.213, LR 0.01
Val set, Accuracy: 0.953700, Loss: 0.149
Iter 1872, 7 / 10, Loss 0.158, LR 0.01
Val set, Accuracy: 0.957200, Loss: 0.138
Iter 2106, 8 / 10, Loss 0.121, LR 0.01
Val set, Accuracy: 0.955200, Loss: 0.149
Iter 2340, 9 / 10, Loss 0.108, LR 0.001
Val set, Accuracy: 0.959800, Loss: 0.134




initial lr:1  batchsize 256  epochs: 10 use_conv: True
Iter 234, 0 / 10, Loss 0.179, LR 1
Val set, Accuracy: 0.952900, Loss: 0.158
Iter 468, 1 / 10, Loss 0.159, LR 1
Val set, Accuracy: 0.958800, Loss: 0.137
Iter 702, 2 / 10, Loss 0.134, LR 1
Val set, Accuracy: 0.961400, Loss: 0.130
Iter 936, 3 / 10, Loss 0.153, LR 0.1
Val set, Accuracy: 0.963800, Loss: 0.121
Iter 1170, 4 / 10, Loss 0.081, LR 0.1
Val set, Accuracy: 0.962900, Loss: 0.120
Iter 1404, 5 / 10, Loss 0.065, LR 0.1
Val set, Accuracy: 0.963400, Loss: 0.123
Iter 1638, 6 / 10, Loss 0.099, LR 0.01
Val set, Accuracy: 0.963200, Loss: 0.121
Iter 1872, 7 / 10, Loss 0.083, LR 0.01
Val set, Accuracy: 0.963500, Loss: 0.125
Iter 2106, 8 / 10, Loss 0.093, LR 0.01
Val set, Accuracy: 0.964200, Loss: 0.118
Iter 2340, 9 / 10, Loss 0.130, LR 0.001
Val set, Accuracy: 0.963300, Loss: 0.124


  """