"""
 Utility file for visualizing the data / loss curves
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
# pylint: disable=line-too-long
# For plotting graphs via ssh with no display
# Ref: https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined


def visualize_data_tsne(Z, labels, num_clusters, title):
    '''
    TSNE visualization of the points in latent space Z
    :param Z: Numpy array containing points in latent space in which clustering was performed
    :param labels: True labels - used for coloring points
    :param num_clusters: Total number of clusters
    :param title: filename where the plot should be saved
    :return: None - (side effect) saves clustering visualization plot in specified location
    '''
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=fig.dpi)


def visualize_data_pca(Z, labels, num_clusters, title):
    '''
    PCA visualization of the points in latent space Z
    :param Z: Numpy array containing points in latent space in which clustering was performed
    :param labels: True labels - used for coloring points
    :param num_clusters: Total number of clusters
    :param title: filename where the plot should be saved
    :return: None - (side effect) saves clustering visualization plot in specified location
    '''
    labels = labels.astype(int)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(principal_components[:, 0], principal_components[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=fig.dpi)


def visualize_plot(x_label, y_label, plot_title, info_train, info_val, epoch, path):
    """
    plots the curve for training and validation loss / accuracy / NMI
    :param x_label: label for x -axis
    :param y_label: label for y-axis
    :param plot_title: plot title
    :param info_train: information about training data
    :param info_val: information about validation data
    :param epoch: number of epochs
    :param path: path where to save image
    :return: None
    """

    fig = plt.figure()
    plt.plot(epoch, info_train)
    plt.plot(epoch, info_val)
    plt.legend(['Train', 'Val'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.grid(True)
    fig.savefig(path, dpi=fig.dpi)
