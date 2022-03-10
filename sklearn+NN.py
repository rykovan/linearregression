#sklearn LR
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline

dataset = pd.read_csv("/Users/natalarykova/Desktop/Howell1.csv", sep=";")

dataset.shape
dataset.head()
dataset.describe()
dataset.plot(x = 'height', y = 'weight', style = 'o')
plt.title('weight vs height')
plt.xlabel("height")
plt.ylabel("weight")
plt.show()
x1 = dataset[["weight"]].to_numpy()
y2 = dataset[["height"]].to_numpy()

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x1, y2)  
print(regressor.intercept_)
print(regressor.coef_)
y2_pred = regressor.predict(x1)
plt.plot(y2, y2_pred, 'bo')

#LR-NN
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
    def forward(self, x): #Who and where is calling that method?
        x = self.flatten(x)
        out = self.linear_layer(x)
        return out

def rescale(v):
    (std, mean)=torch.std_mean(v)
    v = (v - mean)/std
    return v
    
tbl = pd.read_csv("/Users/natalarykova/Desktop/Howell1.csv", sep=";")
x = torch.tensor(tbl[["weight"]].to_numpy(), dtype = torch.float32)
y = torch.tensor(tbl[["height"]].to_numpy(), dtype = torch.float32)
##x=rescale(x)#why comment?
##y=rescale(y)
eta = 1e-5
epochs = 500
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
    err.backward()#??
    opt.step()
    print(f"err:{err:.6}")

print(list(model.named_parameters()))
ypred = model(x).detach().numpy()
plt.scatter(y, ypred)
plt.show()

#plot it

import matplotlib.pyplot as plt

plt.scatter(y2, y2_pred, s=10, c='b', marker="s", label='lr_sklearn')
plt.scatter(y, y_pred, s=10, c='r', marker="o", label='NN_prediction')
plt.legend(loc='upper left')
plt.show()