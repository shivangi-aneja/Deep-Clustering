"""
 init class for all the autoencoders for all the datasets
"""

from deep_clustering.autoencoders.base import BaseAutoencoder
from deep_clustering.autoencoders.mnist_models import *
from deep_clustering.autoencoders.cifar10_models import *
from deep_clustering.autoencoders.svhn_models import *
from deep_clustering.autoencoders.stl10_models import *
from deep_clustering.autoencoders.fmnist_models import *


AUTOENCODERS = {"mnist_autoencoder1", "mnist_autoencoder2", "mnist_autoencoder3", "mnist_autoencoder4", "mnist_autoencoder5",
                "mnist_autoencoder6", "mnist_autoencoder7", "mnist_autoencoder8", "mnist_autoencoder9", "mnist_autoencoder10",
                "mnist_autoencoder11",
                "fmnist_autoencoder1","fmnist_autoencoder2",
                "cifar10_autoencoder1", "cifar10_autoencoder2", "cifar10_autoencoder3", "cifar10_autoencoder4", "cifar10_autoencoder5",
                "cifar10_autoencoder6", "cifar10_autoencoder7",
                "svhn_autoencoder1",
                "stl10_autoencoder1", "stl10_autoencoder2", "stl10_autoencoder3", "stl10_autoencoder4"}

def get_available_autoencoders():
    """
    lists all the available autoencoders
    :return: None
    """
    return sorted(AUTOENCODERS)


def make_autoencoder(name, *args, **kwargs):
    """
    creates the autoencoder based on the name
    :param name: string name of the autoencoder
    :param args: params for the autoenocoder object
    :param kwargs: params for the autoenocoder object
    :return: the autoencoder object
    """
    name = name.strip().lower()
    if not name in AUTOENCODERS:
        raise ValueError("invalid autoencoder architecture: '{0}'".format(name))

    elif name == "mnist_autoencoder1":
        return MNIST_Autoencoder1(*args, **kwargs)

    elif name == "mnist_autoencoder2":
        return MNIST_Autoencoder2(*args, **kwargs)

    elif name == "mnist_autoencoder3":
        return MNIST_Autoencoder3(*args, **kwargs)

    elif name == "mnist_autoencoder4":
        return MNIST_Autoencoder4(*args, **kwargs)

    elif name == "mnist_autoencoder5":
        return MNIST_Autoencoder5(*args, **kwargs)

    elif name == "mnist_autoencoder6":
        return MNIST_Autoencoder6(*args, **kwargs)

    elif name == "mnist_autoencoder7":
        return MNIST_Autoencoder7(*args, **kwargs)

    elif name == "mnist_autoencoder8":
        return MNIST_Autoencoder8(*args, **kwargs)

    elif name == "mnist_autoencoder9":
        return MNIST_Autoencoder9(*args, **kwargs)

    elif name == "mnist_autoencoder10":
        return MNIST_Autoencoder10(*args, **kwargs)

    elif name == "mnist_autoencoder11":
        return MNIST_Autoencoder11(*args, **kwargs)

    elif name == "fmnist_autoencoder1":
        return FMNIST_Autoencoder1(*args, **kwargs)

    elif name == "fmnist_autoencoder2":
        return FMNIST_Autoencoder2(*args, **kwargs)

    elif name == "cifar10_autoencoder1":
        return CIFAR_Autoencoder1(*args, **kwargs)

    elif name == "cifar10_autoencoder2":
        return CIFAR_Autoencoder2(*args, **kwargs)

    elif name == "cifar10_autoencoder3":
        return CIFAR_Autoencoder3(*args, **kwargs)

    elif name == "cifar10_autoencoder4":
        return CIFAR_Autoencoder4(*args, **kwargs)

    elif name == "cifar10_autoencoder5":
        return CIFAR_Autoencoder5(*args, **kwargs)

    elif name == "cifar10_autoencoder6":
        return CIFAR_Autoencoder6(*args, **kwargs)

    elif name == "cifar10_autoencoder7":
        return CIFAR_Autoencoder7(*args, **kwargs)

    elif name == "svhn_autoencoder1":
        return SVHN_Autoencoder1(*args, **kwargs)

    elif name == "stl10_autoencoder1":
        return STL10_Autoencoder1(*args, **kwargs)

    elif name == "stl10_autoencoder2":
        return STL10_Autoencoder2(*args, **kwargs)

    elif name == "stl10_autoencoder3":
        return STL10_Autoencoder3(*args, **kwargs)

    elif name == "stl10_autoencoder4":
        return STL10_Autoencoder4(*args, **kwargs)