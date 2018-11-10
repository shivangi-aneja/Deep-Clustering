"""
autoencoders for stl-10 dataset
"""
import torch.nn as nn
from deep_clustering.autoencoders.base import BaseAutoencoder
from deep_clustering.utils.pytorch_modules import Flatten, Reshape

class STL10_Autoencoder1(BaseAutoencoder):
    """
        STL10_Autoencoder1
    """
    def __init__(self, *args, **kwargs):
        super(STL10_Autoencoder1, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
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

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(18432, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, 18432),
            nn.ReLU(),
            Reshape(512, 6, 6),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

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



class STL10_Autoencoder2(BaseAutoencoder):
    """
        STL10_Autoencoder2
    """
    def __init__(self, *args, **kwargs):
        super(STL10_Autoencoder2, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(4608, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, 4608),
            nn.ReLU(),
            Reshape(128, 6, 6),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, padding=2)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)


class STL10_Autoencoder3(BaseAutoencoder):
    """
        STL10_Autoencoder3
    """
    def __init__(self, *args, **kwargs):
        super(STL10_Autoencoder3, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(4608, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, 4608),
            nn.ReLU(),
            Reshape(128, 6, 6),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)



class STL10_Autoencoder4(BaseAutoencoder):
    """
            STL10_Autoencoder4
            For Pixel Per Cell = 8
    """
    def __init__(self, *args, **kwargs):
        super(STL10_Autoencoder4, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Linear(1307, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(500, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(300, self.latent_dim)


        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Linear(self.latent_dim, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Linear(300, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(500, 1307)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
