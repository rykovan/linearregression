import torchvision
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
from sklearn.decomposition import PCA
from itertools import chain


#get data
root = "./data"
train_set = dset.MNIST(root = root, train = True, transform=torchvision.transforms.ToTensor(),download=True)

x_train = train_set.data.numpy()
y_train = train_set.targets.numpy()

x_train = x_train / 255.0 # normalization 
xt_shape = x_train.shape
print("Initial shape", xt_shape)
xt_flat = x_train.reshape(-1, xt_shape[1]*xt_shape[2])
print("Reshape to", xt_flat.shape)

#PCA with 90% OF VARIANCE - save features = by 10 times compress 
pca = PCA(0.90)
xt_encoded = pca.fit_transform(xt_flat)
print("Initial features: ", xt_flat.shape[1])
print("Encoded features: ", pca.n_components_)

#to print:
#Initial shape (60000, 28, 28)
#Reshape to (60000, 784)
#Initial features:  784
#Encoded features:  87 - we need only 87 components

 

#let's see how is a picture in latency
def pca_latent(dataset):
    dataset_flat = dataset.reshape(-1, dataset.shape[1] * dataset.shape[2])
    return pca.transform(dataset_flat)

def plot_manifold(latent_r, labels=None, alpha=0.5):
    plt.figure(figsize=(10, 10))
    if labels is None:
        plt.scatter(latent_r[:, 0], latent_r[:,1], cmap="tab10", alpha=0.9)
    else:    
        plt.scatter(latent_r[:, 0], latent_r[:, 1], c=labels, cmap="tab10", alpha=0.9)
        plt.colorbar()
    plt.show()
    
latent_r = pca_latent(x_train)
#here we can see that it is impossible to classify the objects in 2-d space
plot_manifold(latent_r, y_train)