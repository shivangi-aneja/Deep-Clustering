"""
    dataset class for all the datasets
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import (Subset, ConcatDataset)

def get_available_datasets():
    """
    gets all the datasets
    :return:
    """
    return sorted(DATASETS)


def make_dataset(name):
    """
    it returns dataset according to its name
    :param name: dataset name
    :return: dataset
    """
    name = name.strip().lower()
    if not name in DATASETS:
        raise ValueError("invalid dataset: '{0}'".format(name))
    elif name == 'mnist':
        return MNIST()
    elif name == 'fmnist':
        return FashionMNIST()
    elif name == 'cifar10':
        return CIFAR10()
    elif name == 'stl10':
        return STL10()
    elif name == 'svhn':
        return SVHN()


class BaseDataset(object):
    """
    base dataset
    """
    def _load(self, dirpath):
        """Download if needed and return the dataset.
        Return format is (`torch.Tensor` data, `torch.Tensor` label) iterable
        of appropriate shape or tuple (train data, test data) of such.
        """
        raise NotImplementedError('`load` is not implemented')

    def load(self, dirpath):
        """
        loads the dataset
        :param dirpath: directory path where dataset is stored
        :return:
        """
        return self._load(os.path.join(dirpath, self.__class__.__name__.lower()))

    def n_classes(self):
        """Get number of classes."""
        raise NotImplementedError('`n_classes` is not implemented')

    def load_full_data(self, dirpath):
        """
                loads the full dataset
                :param dirpath: directory path where dataset is stored
                :return:
                """
        return self._load_full_data(os.path.join(dirpath, self.__class__.__name__.lower()))

    def _load_full_data(self, dirpath):
        """Download if needed and return the dataset.
        Return format is (`torch.Tensor` data, `torch.Tensor` label) iterable
        of appropriate shape or tuple (train data, test data) of such.
        """
        raise NotImplementedError('`_load_full_data` is not implemented')

class MNIST(BaseDataset):
    """
    MNIST dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load_full_data(self, dirpath):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(root=dirpath, train=True, download=True, transform=trans)
        test = datasets.MNIST(root=dirpath, train=False, download=True, transform=trans)
        data_list = list()
        data_list.append(train)
        data_list.append(test)
        total = ConcatDataset(data_list)
        return total


    def _load(self, dirpath):
        # Normalized images
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_mask = range(55000)
        val_mask = range(55000, 60000)
        train_val = datasets.MNIST(root=dirpath, train=True, download=True, transform=trans)
        train = Subset(train_val, train_mask)
        val = Subset(train_val, val_mask)
        test = datasets.MNIST(root=dirpath, train=False, download=True, transform=trans)
        return train, val, test

    def n_classes(self):
        return 10


class FashionMNIST(BaseDataset):
    """
    Fashion MNIST dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load_full_data(self, dirpath):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.FashionMNIST(root=dirpath, train=True, download=True, transform=trans)
        test = datasets.FashionMNIST(root=dirpath, train=False, download=True, transform=trans)
        data_list = list()
        data_list.append(train)
        data_list.append(test)
        total = ConcatDataset(data_list)
        return total


    def _load(self, dirpath):
        # Normalized images
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_mask = range(55000)
        val_mask = range(55000, 60000)
        train_val = datasets.FashionMNIST(root=dirpath, train=True, download=True, transform=trans)
        train = Subset(train_val, train_mask)
        val = Subset(train_val, val_mask)
        test = datasets.FashionMNIST(root=dirpath, train=False, download=True, transform=trans)
        return train, val, test

    def n_classes(self):
        return 10


class CIFAR10(BaseDataset):
    """
    CIFAR-10 dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load(self, dirpath):
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_mask = range(45000)
        val_mask = range(45000, 50000)
        train_val = datasets.CIFAR10(root=dirpath, train=True, download=True, transform=transform_train)
        train = Subset(train_val, train_mask)
        val = Subset(train_val, val_mask)
        test = datasets.CIFAR10(root=dirpath, train=False, download=True, transform=transform_test)
        return train, val, test

    def n_classes(self):
        return 10

    def _load_full_data(self, dirpath):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = datasets.CIFAR10(root=dirpath, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root=dirpath, train=False, download=True, transform=transform)
        data_list = list()
        data_list.append(train)
        data_list.append(test)
        total = ConcatDataset(data_list)
        return total


class STL10(BaseDataset):
    """
    STL-10 dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load(self, dirpath):
        trans = transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        val_mask = range(500)
        train = datasets.STL10(root=dirpath, split='train', download=True, transform=trans)
        val = Subset(train, val_mask)
        test = datasets.STL10(root=dirpath, split='test', download=True, transform=trans)
        return train, val, test

    def n_classes(self):
        return 10

    def unsupervised_data(self, dirpath):
        trans = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        train_unsupervised = datasets.STL10(root=dirpath, split='unlabeled', download=True, transform=trans)
        return train_unsupervised

    def _load_full_data(self, dirpath):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        train = datasets.STL10(root=dirpath, split='train', download=True, transform=transform)
        test = datasets.STL10(root=dirpath, split='test', download=True, transform=transform)
        data_list = list()
        data_list.append(train)
        data_list.append(test)
        total = ConcatDataset(data_list)
        return total


class SVHN(BaseDataset):
    """
    SVHN dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load(self, dirpath):
        trans = transforms.ToTensor()
        train = datasets.SVHN(root=dirpath, split='train', download=True, transform=trans)
        test = datasets.SVHN(root=dirpath, split='test', download=True, transform=trans)
        return train, test

    def n_classes(self):
        return 10


DATASETS = {"mnist", "cifar10", "stl10", "svhn", "fmnist"}



