train_labels = load_labels("dataset/train-labels-idx1-ubyte")
train_images = load_images("dataset/train-images-idx3-ubyte")
val_labels = load_labels("dataset/t10k-labels-idx1-ubyte") # 10000
val_images = load_images("dataset/t10k-images-idx3-ubyte") # 10000, 784