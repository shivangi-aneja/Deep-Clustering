"""
    autoencoders for svhn dataset
"""
import torch.nn as nn

from deep_clustering.autoencoders.base import BaseAutoencoder
from deep_clustering.utils.pytorch_modules import Flatten, Reshape


class SVHN_Autoencoder1(BaseAutoencoder):
    """
        SVHN_Autoencoder1
    """
    def __init__(self, *args, **kwargs):
        super(SVHN_Autoencoder1, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(4096, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            Reshape(256, 4, 4),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()

        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
