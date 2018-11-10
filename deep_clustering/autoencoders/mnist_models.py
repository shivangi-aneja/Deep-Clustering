"""
autoencoders for mnist dataset
"""
import torch.nn as nn

from deep_clustering.autoencoders.base import BaseAutoencoder
from deep_clustering.utils.pytorch_modules import Flatten, Reshape


class MNIST_Autoencoder1(BaseAutoencoder):
    """
        MNIST_Autoencoder1
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder1, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            Flatten(),
            nn.Linear(3136, self.latent_dim),
            #nn.BatchNorm1d(self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, 3136),
            #nn.BatchNorm1d(3136),
            nn.ReLU(),
            Reshape(64, 7, 7),
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)


class MNIST_Autoencoder2(BaseAutoencoder):
    """
            MNIST_Autoencoder2
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder2, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(self.dropout),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(self.dropout),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(3136, self.latent_dim),
            nn.Dropout2d(self.dropout)
        )

    def make_decoder(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, 3136),
            nn.ReLU(),
            Reshape(64, 7, 7),
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data)


class MNIST_Autoencoder3(BaseAutoencoder):
    """
            MNIST_Autoencoder3
    """

    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder3, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(16, 120, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(120)

        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.ConvTranspose2d(120, 16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(16, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(6, 1, kernel_size=5, stride=1, padding=0),
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


class MNIST_Autoencoder4(BaseAutoencoder):
    """
            MNIST_Autoencoder4
    """
    # Gives an accuracy of about 51%
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder4, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)

        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),

            nn.ConvTranspose2d(32, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)


class MNIST_Autoencoder5(BaseAutoencoder):
    """
            MNIST_Autoencoder5
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder5, self).__init__(*args, **kwargs)

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


class MNIST_Autoencoder6(BaseAutoencoder):
    """
        MNIST_Autoencoder6
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder6, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Conv2d(1, 50, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(50),
            nn.ReLU(),

            nn.Conv2d(50, 50, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(5000, 3200),
            nn.ReLU(),
            nn.Linear(3200, 160)

        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Linear(160, 3200),
            nn.ReLU(),
            nn.Linear(3200, 5000),
            Reshape(50, 10, 10),
            nn.MaxUnpool2d(kernel_size=2, stride=2),

            nn.ConvTranspose2d(50, 50, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(50),
            nn.ReLU(),

            nn.ConvTranspose2d(50, 1, kernel_size=5, stride=1, padding=0)
        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data)


class MNIST_Autoencoder7(BaseAutoencoder):
    """
        MNIST_Autoencoder7
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder7, self).__init__(*args, **kwargs)

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



class MNIST_Autoencoder8(BaseAutoencoder):
    """
                MNIST_Autoencoder8
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder8, self).__init__(*args, **kwargs)

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
            nn.Linear(3136, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, self.latent_dim),
        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),

            nn.Linear(self.latent_dim, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, 3136),
            nn.BatchNorm1d(3136),
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


class MNIST_Autoencoder9(BaseAutoencoder):
    """
                MNIST_Autoencoder9
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder9, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(2450, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),

            nn.Linear(self.latent_dim, 2450),
            nn.BatchNorm1d(2450),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            Reshape(50, 7, 7),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(50, 20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(20, 1, kernel_size=5, stride=1, padding=2)

        )

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)


class MNIST_Autoencoder10(BaseAutoencoder):
    """
                MNIST_Autoencoder10
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder10, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(4900, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),

            nn.Linear(self.latent_dim, 4900),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            Reshape(100, 7, 7),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(100, 64, kernel_size=3, stride=1, padding=1),
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



class MNIST_Autoencoder11(BaseAutoencoder):
    """
        MNIST_Autoencoder11
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Autoencoder11, self).__init__(*args, **kwargs)

    def make_encoder(self):
        return nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            Flatten(),
            nn.Linear(6272, 3136),
            nn.ReLU(),
            nn.Linear(3136, self.latent_dim)
        )

    def make_decoder(self):
        return nn.Sequential(

            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),

            nn.Linear(self.latent_dim, 3136),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
            nn.Linear(3136, 6272),
            nn.ReLU(),
            Reshape(32, 14, 14),

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
