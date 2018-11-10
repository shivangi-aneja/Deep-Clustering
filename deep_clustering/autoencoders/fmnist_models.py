"""
autoencoder architectures for fmnist dataset
"""
import torch.nn as nn

from deep_clustering.autoencoders.base import BaseAutoencoder
from deep_clustering.utils.pytorch_modules import Flatten, Reshape



class FMNIST_Autoencoder1(BaseAutoencoder):
    """
        FMNIST_Autoencoder1
    """
    def __init__(self, *args, **kwargs):
        super(FMNIST_Autoencoder1, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            Flatten(),
            nn.Linear(3136, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            nn.Linear(self.latent_dim, 3136),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            Reshape(64, 7, 7),
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)

class FMNIST_Autoencoder2(BaseAutoencoder):
    """
        FMNIST_Autoencoder2
    """
    def __init__(self, *args, **kwargs):
        super(FMNIST_Autoencoder2, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(3136, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),

            nn.Linear(self.latent_dim, 3136),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            Reshape(64, 7, 7),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
