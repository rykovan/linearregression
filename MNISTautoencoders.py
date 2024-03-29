import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from datetime import datetime
import torchvision.transforms as transforms




device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'


#encoder class by subclassing torch.nn.Module
#init method storing layers as an attribute
#forward pass of the network
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)
    
#We do something similar for the Decoder class, ensuring we reshape the output.   

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))    
# FInally, we write an Autoencoder class that combines these two. Note that 
#we could have easily written this entire autoencoder as a single neural network,
#but splitting them in two makes it conceptually clearer.   

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
#Next, we will write some code to train the autoencoder 
#on the MNIST dataset.

def train(autoencoder, data, epochs=100):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()

            total_loss += loss.item()
        print(f'epoch: {epoch + 1}, loss = {total_loss:.4f}')

    return autoencoder



latent_dims = 2
autoencoder = Autoencoder(latent_dims).to(device) # GPU

data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), ((0.3081)))]),
               download=True),
        batch_size=128,
        shuffle=True)

start_time = datetime.now()
autoencoder = train(autoencoder, data)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
            
plot_latent(autoencoder, data)


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    
plot_reconstructed(autoencoder)

    


