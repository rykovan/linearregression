#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
from torch.autograd import Variable
import torch.optim


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_layer = torch.nn.Linear(1, 1)
    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_layer(x)
        return out

def rescale(v):
    (std, mean)=torch.std_mean(v)
    v = (v - mean)/std
    return v
    
tbl = pd.read_csv("Howell1.csv", sep=";")
x = torch.tensor(tbl[["weight"]].to_numpy(), dtype = torch.float32)
y = torch.tensor(tbl[["height"]].to_numpy(), dtype = torch.float32)
##x=rescale(x)
##y=rescale(y)
eta = 1e-5
epochs = 50
model = NeuralNetwork()
opt = torch.optim.SGD(model.parameters(), lr=eta, momentum=0.1)
loss = torch.nn.MSELoss()
"""
model.weight=torch.nn.Parameter(torch.tensor(0.))
model.bias=torch.nn.Parameter(torch.tensor(500.))
"""
"""
plt.scatter(x,y)
plt.show()
"""
for i in range(epochs):
    opt.zero_grad()
    xt = x
    yt = y
    yhat = model(xt)
    print(xt.shape)
    print(yhat.shape)
    print(yt.shape)
    err = loss(yhat, yt )
    err.backward()
    opt.step()
    print(f"err:{err:.6}")

print(list(model.named_parameters()))
ypred = model(x).detach().numpy()
plt.scatter(y, ypred)
plt.show()
