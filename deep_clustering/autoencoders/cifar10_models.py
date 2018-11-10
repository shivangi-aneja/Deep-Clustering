"""
autoencoders for cifar-10 dataset
"""
import torch.nn as nn
from deep_clustering.autoencoders.base import BaseAutoencoder
from deep_clustering.utils.pytorch_modules import Flatten, Reshape


class CIFAR_Autoencoder1(BaseAutoencoder):
    """
    CIFAR_Autoencoder1
    """
    def __init__(self, *args, **kwargs):
        super(CIFAR_Autoencoder1, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(512, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(

            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            Reshape(32, 4, 4),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1)


        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)


class CIFAR_Autoencoder2(BaseAutoencoder):
    """
        CIFAR_Autoencoder2
    """
    def __init__(self, *args, **kwargs):
        super(CIFAR_Autoencoder2, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(2048, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(

            nn.Linear(self.latent_dim, 2048),
            nn.ReLU(),
            Reshape(128, 4, 4),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),


        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)



class CIFAR_Autoencoder3(BaseAutoencoder):
    """
        CIFAR_Autoencoder3
    """
    def __init__(self, *args, **kwargs):
        super(CIFAR_Autoencoder3, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(100),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(100, 150, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(150),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(200),
            nn.Tanh()
        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ConvTranspose2d(200, 150, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(150),
            nn.Tanh(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(150, 100, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(100),
            nn.Tanh(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(100, 3, kernel_size=5, stride=1, padding=2)


        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)



class CIFAR_Autoencoder4(BaseAutoencoder):
    """
            CIFAR_Autoencoder4
    """
    def __init__(self, *args, **kwargs):
        super(CIFAR_Autoencoder4, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512)


        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            Reshape(64, 8, 8),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)


class CIFAR_Autoencoder5(BaseAutoencoder):
    """
            CIFAR_Autoencoder5
            For Pixel Per Cell = 8
    """
    def __init__(self, *args, **kwargs):
        super(CIFAR_Autoencoder5, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Linear(155, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(500, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(200, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Linear(self.latent_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),

            nn.Linear(200, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(500, 155)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)


class CIFAR_Autoencoder6(BaseAutoencoder):
    """
            CIFAR_Autoencoder6
            For Pixel Per Cell = 4
    """
    def __init__(self, *args, **kwargs):
        super(CIFAR_Autoencoder6, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Linear(587, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(200, self.latent_dim)


        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Linear(self.latent_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),

            nn.Linear(200, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(300, 587)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)


class CIFAR_Autoencoder7(BaseAutoencoder):
    """
            CIFAR_Autoencoder7
            For Pixel Per Cell = 8
    """
    def __init__(self, *args, **kwargs):
        super(CIFAR_Autoencoder7, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Linear(155, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(100, self.latent_dim)


        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Linear(self.latent_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(100, 155)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
