import pandas as pd
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt



class Data(Dataset):
    def __init__(self, train=True):

        data=pd.read_csv('') #path to the dataset #NM
        if (train):
            self.x= torch.tensor(data.iloc[0:7903,:].drop([''],axis=1).values, dtype=torch.float) #n 7090 is the number of samples and '' is the y column label #NM
            self.y=torch.tensor(data.loc[0:7903,''], dtype.float).reshape((7904,1)) # n should be about 80% of the data #NM
            self.len=self.x.shape[0]
        else:
            self.x= torch.tensor(data.iloc[7903:,:].drop([''],axis=1).values, dtype=torch.float) #n 7090 is the number of samples and '' is the y column label #NM
            self.y=torch.tensor(data.loc[7903:,''], dtype.float).reshape((1975,1)) #NM
            self.len=self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

data_train =Data(train=True)
train_loader = DataLoader(dataset=data_train, batch_size=5) #NM

class multi_logistic(nn.Module): #softmax
    def __init__(self, inputs, outputs, hidden):
        super(multi_logistic,self).__init__()
        self.linear1=nn.Linear(inputs,hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden,outputs)

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

model = multi_logistic() #need the three, inputs, outputs and hidden layers #NM
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1) #NM

epochs = 100 #NM
LOSS = []
for epoch in range(epochs):
    for x,y in train_loader:
        yhat = model(x)
        loss = criterion(yhat,y)
        optimizer.zero_grad()
        loss.backwards()
        optimizer.step()

        LOSS.append(loss.item())


fig, ax1 = plt.subplot()
color = 'tab:red'
ax1.plot(Loss),color=color)
ax1.xlabel('epoch',color=color)
ax1.ylabel('Cost',color=color)
ax1.tick_params(axis='y',color=color)
