import torch
from numpy.testing import assert_almost_equal

import deep_clustering.utils.tests.env
from deep_clustering.utils.dataset import *

DATA_PATH = 'data/'


def test_mnist():
    dataset = MNIST()
    train_data, test_data = dataset.load(DATA_PATH)
    assert 60000 == len(train_data)
    assert 10000 == len(test_data)

    x, y = train_data[0]
    assert isinstance(x, torch.FloatTensor)
    assert isinstance(y, torch.Tensor)
    assert x.size() == torch.Size([1, 28, 28])
    assert y.size() == torch.Size([])
    assert_almost_equal(0.941176474094, x[0, 14, 14].numpy())

    x, y = test_data[0]
    assert isinstance(x, torch.FloatTensor)
    assert isinstance(y, torch.Tensor)
    assert x.size() == torch.Size([1, 28, 28])
    assert y.size() == torch.Size([])
    assert_almost_equal(0.725490212440, x[0, 7, 7].numpy())
