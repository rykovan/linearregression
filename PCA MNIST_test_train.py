#PCA MNIST
 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
from sklearn.decomposition import PCA
from itertools import chain

USE_CUDA = True   
#get data
root = "./data"
train_set = dset.MNIST(root = root, train = True, transform=torchvision.transforms.ToTensor(),download=True)
test_set = dset.MNIST(root = root, train = False, transform=torchvision.transforms.ToTensor(),download=True)    
    
x_train = train_set.data.numpy()
y_train = train_set.targets.numpy()
x_test = test_set.data.numpy()
y_test = test_set.targets.numpy()

x_train, x_test = x_train / 255.0, x_test / 255.0
xt_shape = x_train.shape
print("Initial shape", xt_shape)
xt_flat = x_train.reshape(-1, xt_shape[1]*xt_shape[2])
print("Reshape to", xt_flat.shape)

#PCA with 90% OF VARIANCE
pca = PCA(0.90)
xt_encoded = pca.fit_transform(xt_flat)
print("Initial features: ", xt_flat.shape[1])
print("Encoded features: ", pca.n_components)

#TEST DATA - dimension definition
xtest_shape = x_test.shape
print("Encoded xtest_decoded shape is" , x_test.shape)
xtest_flat = x_test.reshape(-1, xtest_shape[1]* x_test.shape[2])
xtest_encoded = pca.transform(xtest_flat)
xtest_decoded = pca.inverse_transform(xtest_encoded).reshape(xtest_shape)
print("Decoded.xtest_decoded shape is ", xtest_decoded.shape)


def plot_images(images, title):
    fig = plt.figure(figsize=(16,3))
    columns = images.shape[0]
    rows = 1
    for i in range(columns):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i], cmap="gray_r", clim=(0,1))
    fig.suptitle(title)
    plt.show()
    
    
    
sample_indices = np.random.choice(x_test.shape[0], 6)
samples_orig = x_test[sample_indices]
samples_decoded = xtest_decoded[sample_indices]
                                
plot_images(samples_orig, "Original x_test")
plot_images(samples_decoded, "PCA decoded x_test")   

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
    
latent_r = pca_latent(x_test)
plot_manifold(latent_r, y_test)
    