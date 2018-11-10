"""
base class for autoencoder creation
"""
import torch.nn as nn


class BaseAutoencoder(nn.Module):
    """
    Base Class for autoencoders
    """

    def __init__(self, latent_dim, dropout):
        """
        initialize the parameters
        :param latent_dim: latent dimension of the autoencoder
        :param dropout: dropout rate for the autoencoder
        """
        super(BaseAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()
        self.init()

    def make_encoder(self):
        """
        create the encoder part of the autoencoder
        :return: encoder
        """
        raise NotImplementedError('`make_encoder` is not implemented')

    def make_decoder(self):
        """
        create the decoder part of the autoencoder
        :return: decoder
        """
        raise NotImplementedError('`make_decoder` is not implemented')

    def init(self):
        """
        init method
        :return:
        """
        pass

    def forward(self, x):
        """
        pass input through the encoder and reconstruct with decoder
        :param x: original input
        :return: x_recon : reconstructed input, z : latent representation
        """
        unpool_info = []

        for m in self.encoder:
            if isinstance(m, nn.MaxPool2d):
                output_size = x.size()
                x, pool_idx = m(x)
                unpool_info.append({'output_size': output_size,
                                    'indices': pool_idx})
            else:
                x = m(x)
        z = x

        for m in self.decoder:
            if isinstance(m, nn.MaxUnpool2d):
                x = m(x, **unpool_info.pop())
            else:
                x = m(x)
        x_recon = x

        return x_recon, z
