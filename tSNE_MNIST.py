import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

trainset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
)

dataloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=len(trainset))
images, labels = next(iter(dataloader))
images_flat = images[:, 0].reshape(-1, 784)
images_flat=images_flat[:20000,:]
labels = labels[:20000]

[u,s,v] = torch.pca_lowrank(images_flat)

proj = torch.matmul(images_flat, v[:,:2])
RS = 200
MNIST_tsne = TSNE(random_state=RS).fit_transform(proj)

plt.scatter(MNIST_tsne[:,0], MNIST_tsne[:,1], c=labels, cmap="tab10")
plt.colorbar()
plt.show()