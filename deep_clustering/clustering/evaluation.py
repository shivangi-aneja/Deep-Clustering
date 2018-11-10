"""
 Contains functions used for evaluation of clustering
"""
import numpy as np
from sklearn import metrics
from sklearn.cluster.k_means_ import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn import manifold
from sklearn.decomposition import PCA

# pylint: disable=line-too-long

def evaluate_tsne_clustering(data, labels, num_clusters, k_init, n_components):
    """
     evaluates t-sne clustering on data given
    :param data: input data
    :param labels: ground truth of data
    :param num_clusters: number of clusters
    :param k_init: Number of time the k-means algorithm will be run with different centroid seeds
    :param n_components: Number of components for t-sne visualzation
    :return: accuracy, nmi
    """
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    data_tsne = tsne.fit_transform(data)
    return evaluate_k_means_raw(data=data_tsne, true_labels=labels, n_clusters=num_clusters, k_init=k_init)


def run_pca(data, dimensions):
    """
    runs pca on data given
    :param data: input data
    :param dimensions: number of components selected for principal components
    :return: pca_results
    """
    pca = PCA(n_components=dimensions)
    pca.fit(data)
    pca_results = pca.transform(data)
    return pca_results


def evaluate_k_means_raw(data, true_labels, n_clusters, k_init):
    """
    Clusters data with K-Means algorithm and then returns clustering accuracy and NMI
    :param data: Points that need to be clustered as a numpy array
    :param true_labels: True labels for the given points
    :param n_clusters: Total number of clusters
    :return: ACC, NMI
    """
    # https://github.com/Datamine/MNIST-K-Means-Clustering/blob/master/Kmeans.ipynb
    # http://johnloeber.com/docs/kmeans.html
    # Llyod's Algorithm for K-Means Clustering

    kmeans = KMeans(n_clusters=n_clusters, n_init=k_init)
    kmeans.fit(data)
    acc = cluster_acc(true_labels, kmeans.labels_)
    nmi = metrics.normalized_mutual_info_score(true_labels, kmeans.labels_)
    return acc, nmi


def cluster_acc(y_true, y_pred):
    """
    Uses the hungarian algorithm to find the best permutation mapping and then calculates the accuracy wrt
    Implementation inpired from https://github.com/piiswrong/dec, since scikit does not implement this metric
    this mapping and true labels
    :param y_true: True cluster labels
    :param y_pred: Predicted cluster labels
    :return: accuracy score for the clustering
    """
    D = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((D, D), dtype=np.int32)
    for i in range(y_pred.size):
        idx1 = int(y_pred[i])
        idx2 = int(y_true[i])
        w[idx1, idx2] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluate_kmeans(data, labels, nclusters, method_name):
    """
    Clusters data with kmeans algorithm and then returns the string containing method name and metrics,
    and also the evaluated cluster centers
    :param data: Points that need to be clustered as a numpy array
    :param labels: True labels for the given points
    :param nclusters: Total number of clusters
    :param method_name: Name of the method from which the clustering space originates (only used for printing)
    :return: Formatted string containing metrics and method name, cluster centers
    """
    kmeans = KMeans(n_clusters=nclusters, n_init=20)
    kmeans.fit(data)
    return get_cluster_metric_string(method_name, labels, kmeans.labels_), kmeans.cluster_centers_


def get_cluster_metric_string(method_name, labels_true, labels_pred):
    """
    Creates a formatted string containing the method name and acc, nmi metrics - can be used for printing
    :param method_name: Name of the clustering method (just for printing)
    :param labels_true: True label for each sample
    :param labels_pred: Predicted label for each sample
    :return: Formatted string containing metrics and method name
    """
    acc = cluster_acc(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    return '%-10s     %8.3f     %8.3f' % (method_name, acc, nmi)


def evaluate_kmeans_unsupervised(data, nclusters, k_init=20):
    """
    Clusters data with kmeans algorithm and then returns the cluster centroids
    :param data: Points that need to be clustered as a numpy array
    :param nclusters: Total number of clusters
    :param method_name: Name of the method from which the clustering space originates (only used for printing)
    :return: Formatted string containing metrics and method name, cluster centers
    """
    kmeans = KMeans(n_clusters=nclusters, n_init=k_init)
    kmeans.fit(data)
    return kmeans.cluster_centers_
