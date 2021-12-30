# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.nn as nn
import onnx
from torch.nn.modules.linear import Identity

def conv3x3_bn_relu(in_feature, out_feature, stride = 1, relu = True, bias = True): # 默认有bias
    modules = [
        nn.Conv2d(in_feature, out_feature, kernel_size = 3, stride = stride, padding = 1, bias = bias),
        nn.BatchNorm2d(out_feature)
    ]

    if relu:
        modules.append(nn.ReLU(inplace = True))

    return nn.Sequential(*modules)


# %%

def conv1x1_bn_relu(in_feature, out_feature, stride = 1, relu = True, bias = True):
    modules = [
        nn.Conv2d(in_feature, out_feature, kernel_size = 1, stride = stride, padding = 0, bias = bias), #! 注意padding = 0，不过你debug的时候再发现也不晚。
        nn.BatchNorm2d(out_feature)
    ]

    if relu:
        modules.append(nn.ReLU(inplace = True))

    return nn.Sequential(*modules)


# %%
class BasicBlock(nn.Module):
    expansion = 1 # 膨胀系数

    def __init__(self, in_feature, planes, stride = 1):
        super().__init__()

        self.conv1 = conv3x3_bn_relu(in_feature, planes, stride = stride, bias = False) #! block里只有第一个conv是有stride的
        self.conv2 = conv3x3_bn_relu(planes, planes * self.expansion, relu = False, bias = False) #! 所有的卷积都没有bias 
        self.relu = nn.ReLU(inplace = True)

        if stride == 1:
            # 如果没有下采样
            self.shortcut = nn.Identity()
        else:
            # 如果有下采样
            self.shortcut = conv1x1_bn_relu(in_feature, planes * self.expansion, stride = stride, relu = False)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.shortcut(identity)
        return self.relu(x)


# %%
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.in_feature = 64                  # 定义layer1 的输入通道数，layer0的输出通道数
        self.block = BasicBlock
        self.all_planes = [64, 128, 256, 512] # 定义每个大层的输出通道数
        self.all_blocks = [2, 2, 2, 2]        # 定义每一个大层的block数

        self.layer0 = conv3x3_bn_relu(3, self.in_feature) # 按 cifar10x10的来，所以没有maxpooling，不变大小
        self.layer1 = self.make_layer(self.all_planes[0], 1, self.all_blocks[0])
        self.layer2 = self.make_layer(self.all_planes[1], 2, self.all_blocks[1])
        self.layer3 = self.make_layer(self.all_planes[2], 2, self.all_blocks[2])
        self.layer4 = self.make_layer(self.all_planes[3], 2, self.all_blocks[3])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)


    def make_layer(self, planes, stride, num_block):
        modules = [self.block(self.in_feature, planes, stride)] #! 只有第一个block是有stride的
        self.in_feature = planes * self.block.expansion
        for i in range(num_block - 1):              
            modules.append(self.block(self.in_feature, planes))
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layer0(x)
        print("finish 0")
        x = self.layer1(x)
        print("finish 1")
        x = self.layer2(x)
        print("finish 2")
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


# %%
model = Model(1000)
torch.onnx.export(model, (torch.zeros(1, 3, 224, 224),), "myresnet18.onnx")

