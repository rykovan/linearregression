import torch
import torchvision
from torchvision import datasets
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim  
#dir(datasets)
transform = transforms.Compose([transforms.ToTensor(),])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

#mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size = 10, shuffle = True)
#testloader = torch.utils.data.DataLoader(mnist_testset, batch_size = 10, shuffle = True)

#for data in trainloader:
    #print(data[0][0])
    #print(data[1][0])
    #break

#plt.imshow(data[0][3][0])
#print(data[1][3])    
    
X = mnist_trainset.data.numpy()
print(X)
x_train = X/ 255.0 # normalization 
xt_shape = x_train.shape
print("Initial shape", xt_shape)
xt_flat = x_train.reshape(-1, xt_shape[1]*xt_shape[2])
print("Reshape to", xt_flat.shape)

Y = xt_flat.transpose(1,0)
b = torch.from_numpy(Y)
[u,s,v] = torch.pca_lowrank(b)

#print(f"b:{b.size()}")
#print(u.size())
#print(s.size())
#print(f"v:{v.size()}")

proj=torch.matmul(b,v[:,:2])
print(proj.size())
plt.scatter(proj[:,0], proj[:,1])
plt.show()
