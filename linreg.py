#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

tbl = pd.read_csv("Howell1.csv", sep=";")
x = torch.tensor(tbl[["weight"]].to_numpy(), dtype = torch.float32)
y = torch.tensor(tbl[["height"]].to_numpy(), dtype = torch.float32)
eta = 0.01
epsilon=1e-8
a = torch.tensor(0., requires_grad=True)
b = torch.tensor(0., requires_grad=True)
while(eta > 1e-6):
    yhat = a * x + b
    loss = torch.nn.MSELoss()
    err = loss(y, yhat)
    err.backward()
    with torch.no_grad():
        a1 = a -   eta * a.grad.data
        b1 = b -  eta * b.grad.data
    if (loss(y,a1*x+b1) > err):
        eta = eta/2
        continue
    else:
        eta = 1.1*eta
        a = a1.clone().detach().requires_grad_(True)
        b=  b1.clone().detach().requires_grad_(True)
print(f"a:{a:.4}")
print(f"b:{b:.4}")
print(f"err:{err:.4}")
        #print(f"a1:{a1}")
