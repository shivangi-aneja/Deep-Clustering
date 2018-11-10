"""
 Custom Methods for the Network Layers
"""
import torch
import torch.nn as nn


class Flatten(nn.Module):
    """
    Examples
    --------
    >>> m = Flatten()
    >>> x = torch.randn(32, 10, 5, 3)
    >>> y = m(x)
    >>> y.size()
    torch.Size([32, 150])
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class View(nn.Module):
    """
    Examples
    --------
    >>> x = torch.randn(32, 10, 5, 3)
    >>> y = View(-1)(x)
    >>> y.size()
    torch.Size([4800])
    >>> View(32, -1)(x).size()
    torch.Size([32, 150])
    >>> View(48, 10, 10)(y).size()
    torch.Size([48, 10, 10])
    """
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class Reshape(nn.Module):
    """
    Examples
    --------
    >>> x = torch.randn(32, 10, 5, 3)
    >>> y = Reshape(-1)(x)
    >>> y.size()
    torch.Size([32, 150])
    >>> Reshape(6, 5, 5)(y).size()
    torch.Size([32, 6, 5, 5])
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.size(0), *self.shape)
