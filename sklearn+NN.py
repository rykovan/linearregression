#!/usr/bin/env python

## sklearn import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
## torch import
import torch
import torch.nn
from torch.autograd import Variable
import torch.optim
##

tbl = pd.read_csv("/Users/natalarykova/Desktop/Howell1.csv", sep=";")
x = tbl[["weight"]].to_numpy()
y = tbl[["height"]].to_numpy()

reg = linear_model.LinearRegression()
reg.fit(x, y)
print(reg.coef_)
print(f"intercept:{reg.intercept_}")

yhatlr = reg.predict(x)

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_layer = torch.nn.Linear(1, 1)
    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_layer(x)
        return out

x = torch.tensor(x, dtype = torch.float32)
y = torch.tensor(y, dtype = torch.float32)
eta = 1e-5
epochs = 50
model = NeuralNetwork()
opt = torch.optim.SGD(model.parameters(), lr=eta, momentum=0.1)
loss = torch.nn.MSELoss()
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
yhatnn = model(x).detach().numpy()

## plot

fig, ax = plt.subplots()
ax.scatter(y, yhatlr, s=1, c="orange")
ax.scatter(y, yhatnn, s=1, c='blue')
ax.set_xlabel("observed")
ax.set_ylabel("predicted")
ax.legend()
plt.show()