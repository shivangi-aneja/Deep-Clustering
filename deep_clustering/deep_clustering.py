"""
 This file contains functions for pre-training and fine-tuning
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision.utils import make_grid
from sklearn.cluster.k_means_ import KMeans
from deep_clustering.preprocess.features import (hog_feature, color_histogram_hsv, extract_features)
from deep_clustering.clustering.evaluation import (evaluate_k_means_raw, evaluate_kmeans, evaluate_kmeans_unsupervised)
from deep_clustering.losses.kl_div_loss import ClusterAssignmentHardeningLoss
from deep_clustering.visualization.visualize import (visualize_plot)
from deep_clustering.logging.logger import rootLogger
from deep_clustering.logging.tf_logger import Logger

# pylint: disable=line-too-long

class DeepClustering(object):
    """Class encapsulating training of autoencoder with clustering and non clustering loss.
    Parameters
    ----------
    autoencoder : `torch.nn.Module`
        Autoencoder model.
    optim : `torch.optim`
    optim_kwargs : dict

    """
    def __init__(self, autoencoder, log_path, plot_path, autoencoder_params=None,
                 optim=None, optim_kwargs=None, pretrain_loss_func=None, fine_tune_loss_func=None,
                 pretrain_epoch=10, fintune_epoch=10,
                 cluster_count=10, save_recon_every_iter=50, use_cuda=None,
                 verbose=True, alpha=0.5, k_init=20, n_components=2, visualize=True, preprocess=False, ppc=8):

        """
        Initialize the variables related to autoencoder
        :param autoencoder: autoencoder object
        :param log_path: path where to save log files
        :param plot_path: path where to save plots
        :param autoencoder_params: parameters for autoenocoder like latent space, dropout etc
        :param optim: optimizer used
        :param optim_kwargs: paramters for optimization like learning rate
        :param pretrain_loss_func: loss function for pre-training
        :param fine_tune_loss_func: loss function for fine-tuning
        :param pretrain_epoch: number of epochs to pre-train the network with
        :param fintune_epoch: number of epochs to fine-tune the network with
        :param cluster_count: number of clusters
        :param save_recon_every_iter:
        :param use_cuda: cuda available
        :param verbose: to print
        :param alpha: scaling factor for loss
        :param k_init: number of times to run k-means algo
        :param n_components: number of components for t-sne
        :param visualize: whether to visualize t-sne curves and other plots
        :param preprocess: whether to preprocess images or not
        :param ppc: pixel per cell for Gradient Of Oriented Histogram Calculation
        """

        self.autoencoder = autoencoder
        if autoencoder_params is None or not len(autoencoder_params):
            autoencoder_params = filter(lambda x: x.requires_grad, self.autoencoder.parameters())

        optim = optim or torch.optim.SGD
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-3)
        if optim == torch.optim.LBFGS:
            optim_kwargs = {'lr': optim_kwargs['lr']}
        self.optim = optim(autoencoder_params, **optim_kwargs)
        self.log_path = log_path
        self.plot_path = plot_path
        self.pretrain_loss_func = pretrain_loss_func or nn.MSELoss()
        self.fine_tune_loss_func = fine_tune_loss_func or ClusterAssignmentHardeningLoss()
        self.pretrain_epoch = pretrain_epoch
        self.fine_tune_epoch = fintune_epoch
        self.cluster_count = cluster_count
        self.save_recon_every_iter = save_recon_every_iter
        self.alpha = alpha
        self.k_init = k_init
        self.n_components = n_components

        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.autoencoder.cuda()

        self.verbose = verbose
        self.visualize = visualize
        self.preprocess = preprocess
        self.ppc = ppc

    def pre_train(self, train_loader, pretrain_model_name):
        """
        function for pre-training the network
        :param train_loader: training data
        :param pretrain_model_name: name to save with for pre-train model
        :return: None
        """
        # pylint: disable=too-many-arguments, too-many-locals
        # Training
        # Load the pretrained model if already saved
        try:
            rootLogger.info("Loading pre-trained model")
            checkpoint = torch.load(pretrain_model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
        except:
            rootLogger.info("Pretrained Model Not Fount")

            if self.verbose:
                rootLogger.info('Total training batches: {}'.format(len(train_loader)))
                rootLogger.info("Pre-training Started")
                rootLogger.info("Type           Epochs          ACC               NMI                [Loss]")

        # For matplotlib plotting
        train_acc = np.array([], dtype=float)
        train_nmi = np.array([], dtype=float)
        train_loss = np.array([], dtype=float)
        val_acc = np.array([], dtype=float)
        val_nmi = np.array([], dtype=float)
        val_loss = np.array([], dtype=float)
        name = pretrain_model_name.rsplit('/', 1)[-1]

        # Run pre-training for specified epochs
        for epoch in range(self.pretrain_epoch):
            self.autoencoder.train()
            Z, y = None, None  # latent codes, and labels
            epoch_train_loss = 0.
            pred_sample = list()
            original_sample = list()

            # Checkpoint after 5 epochs
            if epoch > 0 and epoch%5 == 0:
                try:
                    rootLogger.info("Saving the model")
                    torch.save(self.autoencoder.state_dict(), pretrain_model_name + '.pt')
                    rootLogger.info("Pretrain model Saved")
                except:
                    rootLogger.info("Can't save pretrain model")

            for epoch_iter, (x_b, y_b) in enumerate(train_loader):

                if self.preprocess == 'y':
                    num_color_bins = 10  # Number of bins in the color histogram
                    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
                    x_b = extract_features(x_b.numpy(), feature_fns, verbose=True, ppc=self.ppc)

                # Move the images to the device first before computation
                if self.use_cuda:
                    x_b, y_b = x_b.cuda(), y_b.cuda()

                x_b, y_b = Variable(x_b), Variable(y_b)
                self.optim.zero_grad()

                # Predict the reconstructed output and the encoded output
                x_recon, z_b = self.autoencoder(x_b)

                # Store the encoded output for all the batches
                if Z is None:
                    Z = z_b
                else:
                    Z = torch.cat([Z, z_b], dim=0)

                # Also store the complete list of labels
                if y is None:
                    y = y_b
                else:
                    y = torch.cat([y, y_b], dim=0)

                # Just log 24 images
                if len(pred_sample) <= 24:
                    pred_sample.append(x_recon[0])
                    original_sample.append(x_b[0])

                loss = self.pretrain_loss_func(x_recon, x_b)

                epoch_train_loss += loss.item()

                loss.backward()
                self.optim.step()

            avg_loss = epoch_train_loss/len(train_loader)

            Z = Z.view(Z.size(0), -1)
            Z, y = Z.data, y.data
            if self.use_cuda:
                Z, y = Z.cpu(), y.cpu()
            Z, y = Z.numpy(), y.numpy()

            accuracy, nmi = evaluate_k_means_raw(Z, y, self.cluster_count, self.k_init)

            train_acc = np.append(train_acc, accuracy)
            train_nmi = np.append(train_nmi, nmi)
            train_loss = np.append(train_loss, avg_loss)

            if self.verbose:
                rootLogger.info("Train          %d/%d           %8.3f           %8.3f           [%.4f]" % (epoch + 1, self.pretrain_epoch,
                                                                                                           accuracy, nmi, avg_loss))

                # Tensorboard Logging
                logger = Logger(self.log_path)
                # 1. Log training images (image summary)
                rootLogger.info("Logging Images")
                logger.image_summary('Pretrain Reconstructed Image Training', make_grid(torch.stack(pred_sample)).cpu().detach().numpy(),
                                     epoch + 1)
                logger.image_summary('Pretrain Original Image Training', make_grid(torch.stack(original_sample)).cpu().detach().numpy(),
                                     epoch + 1)

                # 2. Log scalar values (scalar summary)
                # info = {'Training Loss': avg_loss, 'Training Accuracy': accuracy, 'Training NMI':nmi}
                rootLogger.info("Logging Training Results")
                logger.scalar_summary('Pretrain Training Loss', avg_loss, epoch + 1)
                logger.scalar_summary('Pretrain Training Accuracy', accuracy, epoch + 1)
                logger.scalar_summary('Pretrain Training NMI', nmi, epoch + 1)

        if self.visualize == 'y':
            visualize_plot('Epoch', 'Accuracy', 'Pretrain Accuracy', train_acc, val_acc, range(self.pretrain_epoch),
                           self.plot_path+name+"_pretrain_accuracy.png")
            visualize_plot('Epoch', 'NMI', 'Pretrain Train NMI', train_nmi, val_nmi, range(self.pretrain_epoch),
                           self.plot_path + name + "_pretrain_nmi.png")
            visualize_plot('Epoch', 'Loss', 'Pretrain Train Loss ', train_loss, val_loss, range(self.pretrain_epoch),
                           self.plot_path + name + "_pretrain_loss.png")

        # Save the model parameters
        try:
            rootLogger.info("Saving the model")
            torch.save(self.autoencoder.state_dict(), pretrain_model_name + '.pt')
            rootLogger.info("Model Saved")
        except:
            rootLogger.info("Error in Saving the model")


    def pre_test(self, test_loader, pretrain_model_name, variable_path_pretrain):
        """
        function to evaluate the results on pre-trained model
        :param test_loader: data loader on which clustering is evaluated
        :param pretrain_model_name: name with which pre-trained model is saved
        :param variable_path_pretrain: path of pre-trained saved model
        :return: None
        """

        # Testing
        Z, y = None, None
        self.autoencoder.eval()

        # Load the parameters of pretrained model
        checkpoint = torch.load(pretrain_model_name + '.pt')
        self.autoencoder.load_state_dict(checkpoint)

        # Evaluate the results on
        with torch.no_grad():
            for (x_test, y_test) in test_loader:

                if self.preprocess == 'y':
                    num_color_bins = 10  # Number of bins in the color histogram
                    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
                    x_test = extract_features(x_test.numpy(), feature_fns, verbose=True, ppc=self.ppc)

                if self.use_cuda:
                    x_test, y_test = x_test.cuda(), y_test.cuda()

                x_recon, z_b = self.autoencoder(x_test)

                # Store the encoded output for all the batches
                if Z is None:
                    Z = z_b
                else:
                    Z = torch.cat([Z, z_b], dim=0)

                # Also store the complete list of labels
                if y is None:
                    y = y_test
                else:
                    y = torch.cat([y, y_test], dim=0)

        Z = Z.view(Z.size(0), -1)
        Z, y = Z.data, y.data
        if self.use_cuda:
            Z, y = Z.cpu(), y.cpu()
        Z, y = Z.numpy(), y.numpy()

        accuracy, nmi = evaluate_k_means_raw(Z, y, self.cluster_count, self.k_init)

        if self.verbose:
            rootLogger.info("Pre-train : Test Accuracy : %8.3f   Test NMI : %8.3f" % (accuracy, nmi))

        try:
            rootLogger.info("Saving the test encoder output")
            np.savetxt(variable_path_pretrain + '_encoder_output.txt', Z)
            np.savetxt(variable_path_pretrain + '_label.txt', y)
            rootLogger.info("Encoder Output saved")
        except:
            rootLogger.info("Error in Saving the Encoder Output")

    def fine_tune(self, train_loader, pretrain_model_name, finetune_model_name,
                  variable_path_pretrain):

        """
        Trains the autoencoder with combined KL-divergence loss and reconstruction loss
        :param train_loader: Training data
        :param pretrain_model_name: Model Name Pretrain
        :param finetune_model_name: Model Name Finetune
        :param variable_path_pretrain: Pretrain model path
        :return: None - (side effect) saves the trained network params and latent space in appropriate location
        """

        # Load the parameters of pretrained model
        try:
            rootLogger.info("Loading fine-tune model")
            checkpoint = torch.load(finetune_model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
        except:
            rootLogger.info("Fine-tune model not found")
            rootLogger.info("Loading pre-train model")
            checkpoint = torch.load(pretrain_model_name+'.pt')
            self.autoencoder.load_state_dict(checkpoint)

        # Find initial cluster centers
        Z_init = np.loadtxt(variable_path_pretrain+'_encoder_output.txt')
        y_init = np.loadtxt(variable_path_pretrain+'_label.txt')
        quality_desc, cluster_centers = evaluate_kmeans(Z_init, y_init, self.cluster_count, 'Initial Accuracy On Pretrain Data')
        if self.verbose:
            rootLogger.info(quality_desc)

        train_acc = np.array([], dtype=float)
        train_nmi = np.array([], dtype=float)
        train_loss = np.array([], dtype=float)

        val_acc = np.array([], dtype=float)
        val_nmi = np.array([], dtype=float)
        val_loss = np.array([], dtype=float)
        name = finetune_model_name.rsplit('/', 1)[-1]
        centroids = None

        if self.verbose:
            rootLogger.info('Total training batches: {}'.format(len(train_loader)))
            rootLogger.info("Starting Fine-Tuning")
            rootLogger.info("Type           Epochs            ACC               NMI             [NC-Loss]           [C-Loss]        [Loss]")

        for epoch in range(self.fine_tune_epoch):
            self.autoencoder.train()
            Z, y = None, None
            epoch_train_loss = 0.
            epoch_c_loss = 0.
            epoch_nc_loss = 0.
            pred_sample = list()
            original_sample = list()

            # Checkpoint after 5 epochs
            if epoch > 0 and epoch % 5 == 0:
                try:
                    rootLogger.info("Saving the model parameters after every 5 epochs")
                    torch.save(self.autoencoder.state_dict(), finetune_model_name + '.pt')
                    rootLogger.info("Model parameters saved")
                except:
                    rootLogger.info("Can't save fine-tune model")
            if epoch == 0:
                centroids = self.get_data_for_kl_loss(encode_output=Z_init, label_list=y_init,
                                                      n_clusters=self.cluster_count)

            for epoch_iter, (x_b, y_b) in enumerate(train_loader):  # Ignore image labels

                if self.preprocess == 'y':
                    num_color_bins = 10  # Number of bins in the color histogram
                    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
                    x_b = extract_features(x_b.numpy(), feature_fns, verbose=True, ppc=self.ppc)

                # Move the images to the device first before computation
                if self.use_cuda:
                    x_b, y_b = x_b.cuda(), y_b.cuda()

                x_b, y_b = Variable(x_b), Variable(y_b)

                self.optim.zero_grad()

                # Predict the reconstructed output and the encoded output
                x_recon, z_b = self.autoencoder(x_b)

                if len(pred_sample) <= 20:
                    pred_sample.append(x_recon[0])
                    original_sample.append(x_b[0])

                # Compute loss for each batch

                # Compute Non Clustering Loss
                nc_loss = self.pretrain_loss_func(x_recon, x_b)

                # Compute Clustering Loss
                z_b = z_b.view(z_b.size(0), -1)

                c_loss = self.fine_tune_loss_func(z_b, centroids)
                loss = self.alpha * c_loss + (1 - self.alpha) * nc_loss

                # Store the encoded output for all the batches
                if Z is None:
                    Z = z_b
                else:
                    Z = torch.cat([Z, z_b], dim=0)

                # Also store the complete list of labels
                if y is None:
                    y = y_b
                else:
                    y = torch.cat([y, y_b], dim=0)

                # The error is the combination of both losses
                epoch_train_loss += loss.item()
                epoch_c_loss += c_loss
                epoch_nc_loss += nc_loss

                loss.backward()
                self.optim.step()

            avg_error = epoch_train_loss/len(train_loader)
            avg_c_error = epoch_c_loss/len(train_loader)
            avg_nc_error = epoch_nc_loss/len(train_loader)
            Z, y = Z.data, y.data
            if self.use_cuda:
                Z, y = Z.cpu(), y.cpu()
            Z, y = Z.numpy(), y.numpy()

            if (epoch+1) % 5 == 0:
                rootLogger.info("Update centroids")
                centroids = self.get_data_for_kl_loss(encode_output=Z, label_list=y,
                                                      n_clusters=self.cluster_count)

            accuracy, nmi = evaluate_k_means_raw(Z, y, self.cluster_count, self.k_init)

            train_acc = np.append(train_acc, accuracy)
            train_nmi = np.append(train_nmi, nmi)
            train_loss = np.append(train_loss, avg_error)

            if self.verbose:
                rootLogger.info("Train          %d/%d           %8.3f           %8.3f           [%.4f]          [%.4f]          [%.4f]" %
                                (epoch + 1, self.fine_tune_epoch, accuracy, nmi, avg_nc_error, avg_c_error, avg_error))

                # Tensorboard Logging
                logger = Logger(self.log_path)
                # 1. Log training images (image summary)
                rootLogger.info("Logging Images")
                logger.image_summary('Finetune Reconstructed Image Training',
                                     make_grid(torch.stack(pred_sample)).cpu().detach().numpy(),
                                     epoch + 1)
                logger.image_summary('Finetune Original Image Training',
                                     make_grid(torch.stack(original_sample)).cpu().detach().numpy(),
                                     epoch + 1)

                # 2. Log scalar values (scalar summary)
                # info = {'Training Loss': avg_loss, 'Training Accuracy': accuracy, 'Training NMI':nmi}
                rootLogger.info("Logging Training Results")
                logger.scalar_summary('Finetune Training Loss Non-Clustering', avg_nc_error, epoch + 1)
                logger.scalar_summary('Finetune Training Loss Clustering', avg_c_error, epoch + 1)
                logger.scalar_summary('Finetune Training Loss', avg_error, epoch + 1)
                logger.scalar_summary('Finetune Training Accuracy', accuracy, epoch + 1)
                logger.scalar_summary('Finetune Training NMI', nmi, epoch + 1)

        if self.visualize == 'y':
            visualize_plot('Epoch', 'Accuracy', 'Fine Tune Accuracy', train_acc, val_acc, range(self.fine_tune_epoch),
                           self.plot_path + name + "_finetune_accuracy.png")
            visualize_plot('Epoch', 'NMI', 'Fine Tune NMI', train_nmi, val_nmi, range(self.fine_tune_epoch),
                           self.plot_path + name + "_finetune_nmi.png")
            visualize_plot('Epoch', 'Loss', 'Fine Tune Loss ', train_loss, val_loss, range(self.fine_tune_epoch),
                           self.plot_path + name + "_finetune_loss.png")

        # Save the model parameters
        try:
            rootLogger.info("Saving the model parameters")
            torch.save(self.autoencoder.state_dict(), finetune_model_name+'.pt')
            rootLogger.info("Fine-tune model saved")
        except:
            rootLogger.info("Can't save fine-tune model")


    def fine_tune_test(self, test_loader, finetune_model_name, variable_path_fine_tune):
        """
        evaluates fine-tune test accuracy
        :param test_loader: data loader on which clustering is evaluated
        :param finetune_model_name: fine-tune model name
        :param variable_path_fine_tune: path where fine-tune model is stored
        :return: None
        """
        # Testing
        Z, y = None, None
        self.autoencoder.eval()

        # Load the parameters of pretrained model
        checkpoint = torch.load(finetune_model_name + '.pt')
        self.autoencoder.load_state_dict(checkpoint)

        with torch.no_grad():
            for (x_test, y_test) in test_loader:

                if self.preprocess == 'y':
                    num_color_bins = 10  # Number of bins in the color histogram
                    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
                    x_test = extract_features(x_test.numpy(), feature_fns, verbose=True, ppc=self.ppc)

                if self.use_cuda:
                    x_test, y_test = x_test.cuda(), y_test.cuda()

                x_recon, z_b = self.autoencoder(x_test)

                # Store the encoded output for all the batches
                if Z is None:
                    Z = z_b
                else:
                    Z = torch.cat([Z, z_b], dim=0)

                # Also store the complete list of labels
                if y is None:
                    y = y_test
                else:
                    y = torch.cat([y, y_test], dim=0)

        Z = Z.view(Z.size(0), -1)
        Z, y = Z.data, y.data
        if self.use_cuda:
            Z, y = Z.cpu(), y.cpu()
        Z, y = Z.numpy(), y.numpy()

        accuracy, nmi = evaluate_k_means_raw(Z, y, self.cluster_count, self.k_init)

        if self.verbose:
            rootLogger.info("After Fine Tune : Test Accuracy : %8.3f   Test NMI : %8.3f" % (accuracy, nmi))

        try:
            rootLogger.info("Saving Test encoder output")
            np.savetxt(variable_path_fine_tune + '_encoder_output.txt', Z)
            np.savetxt(variable_path_fine_tune + '_label.txt', y)
            rootLogger.info("Encoder output Saved")
        except:
            rootLogger.info("Can't save Encoder output")


    def get_data_for_kl_loss(self, encode_output, label_list, n_clusters):
        """
        returns centroids for KL-divergence loss
        :param encode_output: encoder output
        :param label_list: labels for the encoder output
        :param n_clusters: number of clusters
        :return: centroids
        """

        # if self.use_cuda is False:
        #     data = np.copy(encode_output.data)
        #     label = np.copy(label_list.data)
        # else:
        #     data = np.copy(encode_output.data.cpu())
        #     label = np.copy(label_list.data.cpu())

        data = encode_output
        data_len = len(data)

        if data_len < n_clusters:
            n_clusters = data_len

        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=self.k_init)

        # Fitting the input data
        kmeans.fit(data)

        # Centroid values
        centroids = kmeans.cluster_centers_

        if self.use_cuda:
            return Variable(torch.from_numpy(centroids).float().cuda())

        return Variable(torch.from_numpy(centroids).float())

    def pre_train_unsupervised(self, train_loader, pretrain_model_name, variable_path_pretrain):
        """
        unsupervised pre training
        :param train_loader: training data
        :param pretrain_model_name: model name
        :return: None
        """

        # Training
        try:
            rootLogger.info("Loading pre-trained model")
            checkpoint = torch.load(pretrain_model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
        except:
            rootLogger.info("Pretrained Model Not Fount")

            if self.verbose:
                rootLogger.info('Total training batches: {}'.format(len(train_loader)))
                rootLogger.info("Pre-training Started")
                rootLogger.info("Type           Epochs              [Loss]")

        train_loss = np.array([], dtype=float)
        #val_loss = np.array([], dtype=float)
        #name = pretrain_model_name.rsplit('/', 1)[-1]
        num_color_bins = 10  # Number of bins in the color histogram

        for epoch in range(self.pretrain_epoch):
            self.autoencoder.train()
            Z = None  # latent codes
            epoch_train_loss = 0.
            pred_sample = None

            # Checkpoint after 5 epochs
            if epoch > 0 and epoch%5 == 0:
                try:
                    rootLogger.info("Saving the model")
                    torch.save(self.autoencoder.state_dict(), pretrain_model_name + '.pt')
                    rootLogger.info("Pretrain model Saved")
                except:
                    rootLogger.info("Can't save pretrain model")

            for epoch_iter, (x_b, _) in enumerate(train_loader):

                if self.preprocess == 'y':

                    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
                    x_b = extract_features(x_b.numpy(), feature_fns, verbose=True, ppc=self.ppc)

                # Move the images to the device first before computation
                if self.use_cuda:
                    x_b = x_b.cuda()

                x_b = Variable(x_b)
                self.optim.zero_grad()

                # Predict the reconstructed output and the encoded output
                x_recon, z_b = self.autoencoder(x_b)

                # Store the encoded output for all the batches
                if Z is None:
                    Z = z_b
                else:
                    Z = torch.cat([Z, z_b], dim=0)

                # Store the first predicted images
                if pred_sample is None:
                    pred_sample = x_recon[0]
                else:
                    pred_sample = torch.cat([pred_sample, x_recon[0]], dim=0)

                loss = self.pretrain_loss_func(x_recon, x_b)

                epoch_train_loss += loss.item()

                loss.backward()
                self.optim.step()

            avg_loss = epoch_train_loss/len(train_loader)

            train_loss = np.append(train_loss, avg_loss)

            if self.verbose:
                rootLogger.info("Train          %d/%d          [%.4f]" % (epoch + 1, self.pretrain_epoch, avg_loss))

        # Save the model parameters
        try:
            rootLogger.info("Saving the model")
            torch.save(self.autoencoder.state_dict(), pretrain_model_name + '.pt')
            rootLogger.info("Model Saved")
        except:
            rootLogger.info("Error in Saving the model")

        try:
            rootLogger.info("Saving the training encoder output")
            np.savetxt(variable_path_pretrain + '_unsupervised_train_encoder_output.txt', Z)
            rootLogger.info("Encoder Output saved")
        except:
            rootLogger.info("Error in Saving the Encoder Output")

    def fine_tune_unsupervised(self, train_loader, pretrain_model_name, finetune_model_name,
                               variable_path_pretrain, variable_path_finetune):
        """
        :param train_loader: training data
        :param pretrain_model_name: pretrain model name
        :param finetune_model_name: finetune model name
        :return: None
        """

        try:
            rootLogger.info("Loading fine-tune model")
            checkpoint = torch.load(finetune_model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
        except:
            rootLogger.info("Fine-tune model not found")
            rootLogger.info("Loading pre-train model")
            checkpoint = torch.load(pretrain_model_name+'.pt')
            self.autoencoder.load_state_dict(checkpoint)

        #name = finetune_model_name.rsplit('/', 1)[-1]
        Z_init = np.loadtxt(variable_path_pretrain + '_unsupervised_train_encoder_output.txt')
        centroids = evaluate_kmeans_unsupervised(Z_init, self.cluster_count, self.k_init)

        if self.verbose:
            rootLogger.info('Total training batches: {}'.format(len(train_loader)))
            rootLogger.info("Starting Fine-Tuning")
            rootLogger.info("Type           Epochs            [NC-Loss]           [C-Loss]        [Loss]")

        for epoch in range(self.fine_tune_epoch):
            self.autoencoder.train()
            Z = None
            epoch_train_loss = 0.
            epoch_c_loss = 0.
            epoch_nc_loss = 0.

            # Checkpoint after 5 epochs
            if epoch > 0 and epoch % 5 == 0:
                try:
                    rootLogger.info("Saving the model parameters after every 5 epochs")
                    torch.save(self.autoencoder.state_dict(), finetune_model_name + '.pt')
                    rootLogger.info("Model parameters saved")
                except:
                    rootLogger.info("Can't save fine-tune model")

            for epoch_iter, (x_b, _) in enumerate(train_loader):  # Ignore image labels

                if self.preprocess == 'y':
                    num_color_bins = 10  # Number of bins in the color histogram
                    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
                    x_b = extract_features(x_b.numpy(), feature_fns, verbose=True, ppc=self.ppc)

                # Move the images to the device first before computation
                if self.use_cuda:
                    x_b = x_b.cuda()

                x_b = Variable(x_b)

                self.optim.zero_grad()

                # Predict the reconstructed output and the encoded output
                x_recon, z_b = self.autoencoder(x_b)

                # Compute loss for each batch

                # Compute Non Clustering Loss
                nc_loss = self.pretrain_loss_func(x_recon, x_b)

                # Compute Clustering Loss
                z_b = z_b.view(z_b.size(0), -1)

                c_loss = self.fine_tune_loss_func(z_b, centroids)
                loss = self.alpha * c_loss + (1 - self.alpha) * nc_loss

                # Store the encoded output for all the batches
                if Z is None:
                    Z = z_b
                else:
                    Z = torch.cat([Z, z_b], dim=0)

                # The error is the combination of both losses
                epoch_train_loss += loss.item()
                epoch_c_loss += c_loss
                epoch_nc_loss += nc_loss

                loss.backward()
                self.optim.step()

            avg_error = epoch_train_loss/len(train_loader)
            avg_c_error = epoch_c_loss/len(train_loader)
            avg_nc_error = epoch_nc_loss/len(train_loader)
            Z = Z.data
            if self.use_cuda:
                Z = Z.cpu()
            Z = Z.numpy()

            if (epoch+1) % 5 == 0:
                rootLogger.info("Update centroids")
                centroids = self.get_data_for_kl_loss(encode_output=Z, label_list=None,
                                                      n_clusters=self.cluster_count)

            if self.verbose:
                rootLogger.info("Train          %d/%d          [%.4f]          [%.4f]          [%.4f]" %
                                (epoch + 1, self.fine_tune_epoch, avg_nc_error, avg_c_error, avg_error))

                # Tensorboard Logging
                logger = Logger(self.log_path)

                # 2. Log scalar values (scalar summary)
                # info = {'Training Loss': avg_loss, 'Training Accuracy': accuracy, 'Training NMI':nmi}
                rootLogger.info("Logging Training Results")
                logger.scalar_summary('Finetune Training Loss Non-Clustering', avg_nc_error, epoch + 1)
                logger.scalar_summary('Finetune Training Loss Clustering', avg_c_error, epoch + 1)
                logger.scalar_summary('Finetune Training Loss', avg_error, epoch + 1)


        # Save the model parameters
        try:
            rootLogger.info("Saving the model parameters")
            torch.save(self.autoencoder.state_dict(), finetune_model_name+'.pt')
            rootLogger.info("Fine-tune model saved")
        except:
            rootLogger.info("Can't save fine-tune model")

        try:
            rootLogger.info("Saving encoder output")
            np.savetxt(variable_path_finetune + '_unsupervised_train_encoder_output.txt', Z)
            rootLogger.info("Encoder output Saved")
        except:
            rootLogger.info("Can't save Encoder output")
