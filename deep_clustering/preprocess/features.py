"""
Feature Extraction Helper Functions.
"""
import numpy as np
from skimage.feature import CENSURE
from scipy.ndimage import uniform_filter
import matplotlib
import torch

def extract_features(imgs, feature_fns, verbose=False, ppc=8):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x H X W X C tensor of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    A tensor of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    imgs = np.transpose(imgs, (0, 2, 3, 1))
    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        if feature_fn == hog_feature:
            feats = feature_fn(imgs[0].squeeze(), ppc)
        else:
            feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            if feature_fn == hog_feature:
                imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze(), ppc)
            else:
                imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        # if verbose and i % 1000 == 0:
        #     print('Done extracting features for {}/{} images'.format(
        #         i, num_images))

    # Preprocessing: Subtract the mean feature
    #mean_feat = np.mean(imgs_features, axis=0, keepdims=True)
    #imgs_features -= mean_feat

    # Preprocessing: Divide by standard deviation. This ensures that each feature
    # has roughly the same scale.
    #std_feat = np.std(imgs_features, axis=0, keepdims=True)
    #imgs_features /= std_feat

    # Preprocessing: Add a bias dimension
    imgs_features = np.hstack([imgs_features, np.ones((imgs_features.shape[0], 1))])

    return torch.from_numpy(imgs_features).float()


def rgb2gray(rgb):
    """Convert RGB image to grayscale

      Parameters:
        rgb : RGB image

      Returns:
        gray : grayscale image

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def censure_feature(im):
    """
        CENSURE Feature Extraction
    :param im: image
    :return: image keypoints
    """
    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.atleast_2d(im)

    censure = CENSURE()
    censure.detect(image)
    # censure.scales
    # Fetch only 10 keypoints
    censure_array = np.zeros(10)
    if len(censure.keypoints.ravel()) > 10:
        censure_array = censure.keypoints.ravel()[:10]
    else:
        censure_array[:len(censure.keypoints.ravel())] = censure.keypoints.ravel()
    return censure_array




def hog_feature(im, ppc):
    """Compute Histogram of Gradient (HOG) feature for an image

         Modified from skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : an input grayscale or rgb image

      Returns:
        feat: Histogram of Gradient (HOG) feature

    """
    # pylint: disable=too-many-locals
    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.atleast_2d(im)

    sx, sy = image.shape # image size
    orientations = 9 # number of gradient bins
    cx, cy = (ppc, ppc) # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        temp_mag = np.where(temp_ori > 0, grad_mag, 0)
        uni_fil = uniform_filter(temp_mag, size=(cx, cy))
        orientation_histogram[:, :, i] = uni_fil[cx // 2::cx, cy // 2::cy].T

    return orientation_histogram.ravel()

def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
      1D vector of length nbin giving the color histogram over the hue of the
      input image.
    """
    bins = np.linspace(xmin, xmax, nbin+1)
    hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
    im_hist, bin_edges = np.histogram(hsv[:, :, 0],
                                      bins=bins,
                                      density=normalized)
    im_hist = im_hist * np.diff(bin_edges)

    # return histogram
    return im_hist
