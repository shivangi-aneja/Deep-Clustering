"""
custom loss implementation for k-means loss
"""
import torch
import numpy as np

class KMeansClusteringLoss(torch.nn.Module):

    """
    K-means loss implementaion used for training
    """
    def __init__(self):
        super(KMeansClusteringLoss, self).__init__()

    def forward(self, encode_output, centroids):
        """
        trains the network on k-means clustering loss
        :param encode_output: out of the encoder on which we calcuate loss
        :param centroids: centroids for the encoder output after k-means clustering is done
        :return: k-means loss
        """

        # encode_output: N * d
        # Centroids: K * d
        #
        # N = Batch Size / No.of data points
        #
        # d = dimensionality of data
        #
        # K = Number of clusters

        assert encode_output.size(1) == centroids.size(1), "Dimension mismatch"
        return ((encode_output[:, np.newaxis] - centroids[np.newaxis]) ** 2).sum(2).min(1)[0].mean()

        # Implementation Similar to Old Lasagne Code
        # assert (encode_output.shape[1] == centroids.shape[1]),"Dimensions Mismatch"
        # n = encode_output.shape[0]
        # d = encode_output.shape[1]
        # k = centroids.shape[0]
        #
        # z = encode_output.reshape(n,1,d)
        # z = z.repeat(1,k,1)
        #
        # mu = centroids.reshape(1,k,d)
        # mu = mu.repeat(n,1,1)
        #
        # dist = (z-mu).norm(2,dim=2).reshape((n,k))
        # loss = dist.min(dim=1)[0].mean()
        #
        # return loss
