import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
import torch.optim as optim

from temp.utils import lr_schedule_cosine

data = pickle.load(open("db.pkl", "rb"))
db = data["db"]
caption = data["caption"]


train = db[:80]
test  = db[80:]
len(train), len(test)

# make up train_dataset
lmks_list_train = []
labels_list_train = []

for item in train:
    x = item[4].reshape(1,136).astype(np.float32)
    y = np.array(item[5:]).reshape(1, -1).astype(np.float32)
    lmks_list_train.append(x)
    labels_list_train.append(y)


# make up test_dataset
lmks_list_test = []
labels_list_test = []

for item in test:
    x = item[4].reshape(1,136).astype(np.float32)
    y = np.array(item[5:]).reshape(1, -1).astype(np.float32)
    lmks_list_test.append(x)
    labels_list_test.append(y)

########################################################################

lmks_train = np.concatenate(lmks_list_train, axis = 0)
labels_train = np.concatenate(labels_list_train, axis = 0)

lmks_test = np.concatenate(lmks_list_test, axis = 0)
labels_test = np.concatenate(labels_list_test, axis = 0)

################ mean  and variance #######################################################

# lmks_train_mean   = lmks_train.mean()
# lmks_train_var    = lmks_train.var()

# labels_train_mean = labels_train.mean(axis = 0).reshape(1,-1)
# labels_train_var  = labels_train.var(axis = 0).reshape(1,-1)

#################  min max  ##########################################


min_label_train = np.min(labels_train, axis = 0)
max_label_train = np.max(labels_train, axis = 0) 
max_min_train_label = max_label_train - min_label_train

#####################################################################

norm_train_data = (lmks_train - 256) / 256 
norm_train_label = (labels_train - min_label_train) / max_min_train_label

norm_test_data = (lmks_test - 256) / 256
norm_test_label  = (labels_test - min_label_train) / max_min_train_label


#region
# # norm_train_data  =  (lmks_train - lmks_train_mean) / lmks_train_var
# norm_train_data = (lmks_train - 256) / 256 
# norm_train_label = (labels_train - labels_train_mean) / labels_train_var

# # norm_test_data   = (lmks_test - lmks_train_mean) / lmks_train_var
# norm_test_data = (lmks_test - 256) / 256
# norm_test_label  = (labels_test - labels_train_mean) / labels_train_var
#endregion


def evaluate(test_loader, model, epoch = None, loss = None):
    model.eval()
    for norm_test_data, norm_test_label in test_loader:
        norm_test_data = norm_test_data.to(device)
        norm_test_label = norm_test_label.to(device)
        test_loss = model(norm_test_data, norm_test_label)
        print(f"     test_loss: {test_loss}")
    


class Dataset:
    def __init__(self, norm_data, norm_label):
        self.norm_data = norm_data
        self.norm_label = norm_label
    
    def __getitem__(self, index):
        x = self.norm_data[index]
        y = self.norm_label[index]        

        return x, y
    
    def __len__(self):
         return len(self.norm_data)

train_dataset = Dataset(norm_train_data, norm_train_label)
test_dataset  = Dataset(norm_test_data, norm_test_label)

class Model(nn.Module):
    def __init__(self, infeature, outfeature):
        super().__init__()
        nhidden1 = 256
        nhidden2 = 256
        
        self.layers = nn.Sequential(
            nn.Linear(infeature, nhidden1),
            nn.ReLU(),
            nn.Linear(nhidden1, nhidden2),
            nn.ReLU(),
            nn.Linear(nhidden2, outfeature)
        )
        
        self.loss = nn.MSELoss()

    def init_weights(self,m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

    def inference(self, x):
        return self.layers(x)
        
    def forward(self, x, gt):
        return self.loss(self.inference(x), gt)

device = "cuda:3"
model = Model(136, 8).to(device)
model.apply(model.init_weights)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
test_loader =  torch.utils.data.DataLoader(test_dataset,  batch_size=17, shuffle=False, pin_memory=True, num_workers=0)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 1000
batch_size = 20




for epoch in range(num_epochs):
    model.train()

    train_loss = []
    

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    
    print(f"epoch :{epoch} train_loss:{np.array(train_loss).mean()}")
    evaluate(test_loader, model, epoch)
    print("\n")
        
    

    

