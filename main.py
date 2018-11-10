#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Main file to train and evaluate the models
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from deep_clustering.deep_clustering import DeepClustering
from deep_clustering.autoencoders import (get_available_autoencoders,
                                          make_autoencoder)
from deep_clustering.utils import (get_available_datasets,
                                   make_dataset, RNG)
from deep_clustering.utils.pytorch_dataset_utils import DatasetIndexer
from deep_clustering.losses.kl_div_loss import ClusterAssignmentHardeningLoss
from deep_clustering.losses.k_means_loss import KMeansClusteringLoss
from deep_clustering.logging.logger import rootLogger
from deep_clustering.clustering.evaluation import evaluate_tsne_clustering
from deep_clustering.visualization.visualize import visualize_data_tsne

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad,
    'sgd': torch.optim.SGD,
    'rms_prop': torch.optim.RMSprop,
    'lbgfs' : torch.optim.LBFGS
}

LOSS_FUNCS_PRETRAIN = {
    'mse': nn.MSELoss()
}

LOSS_FUNCS_FINE_TUNE = {
    'kl_div': ClusterAssignmentHardeningLoss(),
    'k_means_loss': KMeansClusteringLoss()
}

LOG_PATH = os.path.join(os.getcwd(), 'logs/')
PLOT_PATH = os.path.join(os.getcwd(), 'plots/')

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general
parser.add_argument('-d', '--dataset', type=str, default='mnist',
                    help="dataset, {'" +\
                         "', '".join(get_available_datasets()) +\
                         "'}")
parser.add_argument('--data-dirpath', type=str, default='data/',
                    help='directory for storing downloaded data')
parser.add_argument('--n-workers', type=int, default=2,
                    help='how many threads to use for I/O')
parser.add_argument('--gpu', type=str, default='0',
                    help="ID of the GPU to train on (or '' to train on CPU)")
parser.add_argument('-rs', '--random-seed', type=int, default=1,
                    help="random seed for training")
parser.add_argument('-tf', '--tf_logs', type=str, default='tf_logs',
                    help="log folder for tensorflow logging")

# autoencoder-related
parser.add_argument('-a', '--autoencoder', type=str, default='mnist_autoencoder1',
                    help="autoencoder architecture name, {'" + \
                         "', '".join(get_available_autoencoders()) +\
                         "'}")
parser.add_argument('-pl', '--pretrain_loss', type=str, default='mse',
                    help="Loss function for pre-training,  {'" + \
                         "', '".join(LOSS_FUNCS_PRETRAIN.keys()) +\
                         "'}")
parser.add_argument('-fl', '--finetune_loss', type=str, default='kl_div',
                    help="loss function for fine-tuning,  {'" + \
                         "', '".join(LOSS_FUNCS_FINE_TUNE.keys()) +\
                         "'}")
parser.add_argument('-nz', '--latent-dim', type=int, default=32,
                    help='latent space dimension for autoencoder')

parser.add_argument('-b', '--batch-size', type=int, default=256,
                    help='input batch size for training')

parser.add_argument('-pe', '--pretrain_epochs', type=int, default=5,
                    help='number of epochs')

parser.add_argument('-fe', '--finetune_epochs', type=int, default=5,
                    help='number of epochs')

parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                    help='initial learning rate')

parser.add_argument('-wd', '--weight-decay', type=float, default=0,
                    help='weight decay')

parser.add_argument('-dp', '--dropout', type=float, default=0,
                    help='dropout')

parser.add_argument('-opt', '--optim', type=str, default='adam',
                    help="optimizer, {'" + \
                         "', '".join(OPTIMIZERS.keys()) +\
                         "'}")

parser.add_argument('-p', '--pretrain_model_name', type=str,
                    default='pretain_model',
                    help='name for pretrain model parameters')

parser.add_argument('-f', '--finetune_model_name', type=str,
                    default='finetune_model',
                    help='name for finetune model parameters')

# clustering-related
parser.add_argument('-alpha', '--alpha', type=float, default=0.5,
                    help='alpha for joint training with clustering/non-clustering loss')
parser.add_argument('-k', '--k_init', type=int, default=30,
                    help="No. of times k-means clustering on the encoder output is run")
parser.add_argument('-c', '--n_components', type=int, default=3,
                    help="No. of components for dimensionality reduction in T-SNE")

# visualization-related
parser.add_argument('-v', '--visualize', type=str, default='n',
                    help=' (y/n) whether to visualize t-sne for the data and encoder output or not')
parser.add_argument('-mp', '--max_points', type=int, default=10000,
                    help="No. of points to plot during t-sne visualization")

# preprocessing-related
parser.add_argument('-pp', '--preprocess', type=str, default='n',
                    help=' (y/n) whether to pre-process the image to calcualate histogram of oriented gradients and color histogram')
parser.add_argument('-ppc', '--pixel_per_cell', type=int, default=8,
                    help=' pixel per cell to calcualate histogram of oriented gradients')

# unsupervised-training-related
parser.add_argument('-ut', '--unsupervised_train', type=str, default='n',
                    help=' whether to train in an unsupervised setting or not.')


# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# print arguments
rootLogger.info("Running with the following parameters:")
rootLogger.info(vars(args))


def main(args=args):
    """
    main function that parses the arguments and trains
    :param args: arguments related
    :return: None
    """
    # pylint: disable=line-too-long
    # load and shuffle data
    dataset = make_dataset(args.dataset)

    train_dataset, val_dataset, test_dataset = dataset.load(args.data_dirpath)
    total_dataset = dataset.load_full_data(args.data_dirpath)

    # For visualization
    max_points = args.max_points
    x_test, y_test = zip(*Subset(test_dataset, range(max_points)))

    rng = RNG(args.random_seed)
    train_ind = rng.permutation(len(train_dataset))
    val_ind = rng.permutation(len(val_dataset))
    test_ind = rng.permutation(len(test_dataset))
    total_ind = rng.permutation(len(total_dataset))

    train_dataset = DatasetIndexer(train_dataset, train_ind)
    val_dataset = DatasetIndexer(val_dataset, val_ind)
    test_dataset = DatasetIndexer(test_dataset, test_ind)
    total_dataset = DatasetIndexer(total_dataset, total_ind)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.n_workers)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.n_workers)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.n_workers)
    total_loader = DataLoader(dataset=total_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.n_workers)


    # build autoencoder model
    autoencoder = make_autoencoder(name=args.autoencoder, latent_dim=args.latent_dim,
                                   dropout=args.dropout)

    # get optimizer
    optim = OPTIMIZERS.get(args.optim, None)
    if not optim:
        raise ValueError("invalid optimizer: '{0}'".format(args.optim))

    # get loss function
    pretrain_loss_func = LOSS_FUNCS_PRETRAIN.get(args.pretrain_loss, None)
    if not pretrain_loss_func:
        raise ValueError("Invalid pretrain loss function: '{0}'".format(args.pretrain_loss))

    finetune_loss = LOSS_FUNCS_FINE_TUNE.get(args.finetune_loss, None)
    if not finetune_loss:
        raise ValueError("Invalid fine-tune loss function: '{0}'".format(args.finetune_loss))

    cluster_count = dataset.n_classes()
    arch_num = args.autoencoder[-1]
    pretrain_model_name = "models/"+args.dataset+"/arch"+arch_num+"/nz"+str(args.latent_dim)+"/"+args.pretrain_model_name
    finetune_model_name = "models/"+args.dataset+"/arch"+arch_num+"/nz"+str(args.latent_dim)+"/"+args.finetune_model_name
    variable_path_pretrain = "vars/"+ args.dataset+"/arch"+arch_num+"/nz"+str(args.latent_dim)+"/"+args.pretrain_model_name
    variable_path_fine_tune = "vars/" + args.dataset +"/arch"+arch_num+"/nz"+str(args.latent_dim)+ "/" + args.finetune_model_name
    plot_path = PLOT_PATH+args.dataset+"/arch"+arch_num+"/nz"+str(args.latent_dim)+"/"
    alpha = args.alpha
    k_init = args.k_init
    n_components = args.n_components
    vis = args.visualize

    # Create Autoencoder according to params
    model = DeepClustering(autoencoder=autoencoder, log_path=LOG_PATH+args.tf_logs+"/", plot_path=plot_path,
                           optim=optim,
                           optim_kwargs={'lr': args.learning_rate, 'weight_decay':args.weight_decay},
                           pretrain_loss_func=pretrain_loss_func, fine_tune_loss_func=finetune_loss,
                           pretrain_epoch=args.pretrain_epochs,
                           fintune_epoch=args.finetune_epochs, cluster_count=cluster_count, use_cuda=bool(args.gpu),
                           alpha=alpha, k_init=k_init, n_components=n_components, verbose=True, visualize=vis,
                           preprocess=args.preprocess, ppc=args.pixel_per_cell)



    # # Used for Unsupervised clustering
    # unsupervised_train = args.unsupervised_train
    # if args.dataset == "stl10" and unsupervised_train == 'y':
    #     unsupervised_dataset = dataset.unsupervised_data(args.data_dirpath)
    #     unsupervised_ind = rng.permutation(len(unsupervised_dataset))
    #     unsupervised_dataset = DatasetIndexer(unsupervised_dataset, unsupervised_ind)
    #     unsupervised_loader = DataLoader(dataset=unsupervised_dataset,
    #                                      batch_size=args.batch_size,
    #                                      shuffle=True,
    #                                      num_workers=args.n_workers)
    #     if args.pretrain_epochs > 0:
    #         model.pre_train_unsupervised(train_loader=unsupervised_loader, pretrain_model_name=pretrain_model_name,
    #                                      variable_path_pretrain=variable_path_pretrain)
    #
    #     if args.finetune_epochs > 0:
    #         model.fine_tune_unsupervised(train_loader=unsupervised_loader, pretrain_model_name=pretrain_model_name,
    #                                      finetune_model_name=finetune_model_name,
    #                                      variable_path_pretrain=variable_path_pretrain,
    #                                      variable_path_finetune=variable_path_fine_tune)

    # Supervised Train, Test
    # Pretrain
    if args.pretrain_epochs > 0:
        model.pre_train(train_loader=total_loader, pretrain_model_name=pretrain_model_name)
    model.pre_test(test_loader=total_loader, pretrain_model_name=pretrain_model_name,
                   variable_path_pretrain=variable_path_pretrain)

    # finetune
    if args.finetune_epochs > 0:
        model.fine_tune(train_loader=total_loader, pretrain_model_name=pretrain_model_name,
                        variable_path_pretrain=variable_path_pretrain,
                        finetune_model_name=finetune_model_name)
    # Test
    model.fine_tune_test(test_loader=total_loader, finetune_model_name=finetune_model_name,
                         variable_path_fine_tune=variable_path_fine_tune)

    # T-SNE of Original Pixel Space
    if vis == 'y':
        # T-SNE of Pretrained Encoder
        try:

            rootLogger.info("Original T-SNE on Test Data")
            x_test = torch.cat(x_test).view(len(x_test), -1).numpy()
            y_test = torch.stack([torch.Tensor(y_test)], dim=0).numpy().ravel()

            acc, nmi = evaluate_tsne_clustering(data=x_test, labels=y_test, num_clusters=dataset.n_classes(),
                                               k_init=k_init, n_components=n_components)
            rootLogger.info("Accuracy : %8.3f    NMI : %8.3f " % (acc, nmi))


            #rootLogger.info("Original Test data t-sne")

            #visualize_data_tsne(x_test[0:max_points], y_test[0:max_points], cluster_count, plot_path + "/original_tsne.png")
            #visualize_data_pca(x_test, y_test, cluster_count, plot_path + "/original_test_pca.png")


            rootLogger.info("Pretrain Test data t-sne")
            z_pretest = np.loadtxt(variable_path_pretrain + '_encoder_output.txt')
            y_pretest = np.loadtxt(variable_path_pretrain + '_label.txt')
            acc, nmi = evaluate_tsne_clustering(data=z_pretest[0:max_points], labels=y_pretest[0:max_points], num_clusters=dataset.n_classes(),
                                                k_init=k_init, n_components=n_components)
            rootLogger.info("Accuracy : %8.3f    NMI : %8.3f " % (acc, nmi))
            print("Visualizing")
            visualize_data_tsne(z_pretest[0:max_points], y_pretest[0:max_points], cluster_count,
                                plot_path + args.pretrain_model_name + "_tsne.png")

            rootLogger.info("Finetune Test data t-sne")
            z_finetest = np.loadtxt(variable_path_fine_tune + '_encoder_output.txt')
            y_finetest = np.loadtxt(variable_path_fine_tune + '_label.txt')
            acc, nmi = evaluate_tsne_clustering(data=z_finetest[0:max_points], labels=y_finetest[0:max_points], num_clusters=dataset.n_classes(),
                                                k_init=k_init, n_components=n_components)
            rootLogger.info("Accuracy : %8.3f    NMI : %8.3f " % (acc, nmi))


            print("Visualizing")
            visualize_data_tsne(z_finetest[0:max_points], y_finetest[0:max_points], cluster_count,
                                plot_path + args.finetune_model_name + "_tsne.png")


        except:
            print("T-SNE could not be plotted for Encoder")


if __name__ == '__main__':
    main()
