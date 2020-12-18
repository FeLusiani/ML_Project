import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from collections import OrderedDict
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import load_monk_data, preprocess_monk


df = load_monk_data(1)
X, Y = preprocess_monk(df)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


class NN_BinClassifier(torch.nn.Module):
    def __init__(self, layers):
        super(NN_BinClassifier, self).__init__()

        net_topology = OrderedDict()
        for i in range(len(layers)):
            if i != len(layers)-1:
                net_topology[f'fc{i}'] = torch.nn.Linear(layers[i], layers[i+1])
                net_topology[f'LeakyReLu{i}'] = torch.nn.LeakyReLU()
            else:
                # last layer (output)
                net_topology[f'fc{i}'] = torch.nn.Linear(layers[i], 1)
                net_topology[f'Sigmoid'] = torch.nn.Sigmoid()
        
        self.net = torch.nn.Sequential(net_topology)

    def forward(self, x):
        out = self.net(x)
        return out


net = NN_BinClassifier([17, 7, 2])
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(3000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(X_train.shape[0]):
        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.Tensor(X_train[i:min(i+10,X_train.shape[0]),:])
        labels = torch.Tensor(y_train[i:min(i+10,X_train.shape[0])].reshape(-1,1))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # print epoch statistics
    
    if i % 1 == 0:    # print every 1 epochs
        print(f'epoch {epoch} \t loss: {running_loss :.3f}')

print('Finished Training')