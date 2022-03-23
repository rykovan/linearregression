import matplotlib.pyplot as plt
import time
import torch
import torchvision.datasets
from sklearn.manifold import TSNE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
##print(device)

def print_device(name,v):
    print(f"{name} is in device:{v.device}")

start=time.time()




mnist = list(torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor()))

mnistarray = torch.cat(
    [mnist[i][0].squeeze().flatten().unsqueeze(1) for i in range(len(mnist))], 1
)

mnistlabels = [mnist[i][1] for i in range(len(mnist))]

#mnistarray=mnistarray.to('cpu')
mnistarray=mnistarray.to('cuda')
mnistarray = torch.transpose(mnistarray, 0, 1)
print_device("mnistarray", mnistarray)
[u,s,v]= torch.pca_lowrank(mnistarray)
proj=torch.matmul(mnistarray, v[:,:2])
#print_device("proj", proj)
#print(proj.size())
proj=proj.to('cpu')
#print_device("proj", proj)

RS = 123
MNIST_tsne = TSNE(random_state=RS).fit_transform(proj[:1000,:])
mnistlabels = mnistlabels[:1000]
stop=time.time()
print(f"elapsed:{stop-start:.3f}")


plt.scatter(MNIST_tsne[:,0], MNIST_tsne[:,1], c=mnistlabels, cmap="tab10")
plt.colorbar()
plt.show()

