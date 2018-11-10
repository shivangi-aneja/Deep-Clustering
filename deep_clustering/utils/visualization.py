"""
Utility class to save the images
"""
import os
import torchvision.utils as vutils


def save_image(tensor_minibatch, image_path, file_name):
    """
    saves the image batch passed
    :param tensor_minibatch:  minibatch of images
    :param image_path:  image path
    :param file_name: file name
    :return: None
    """
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    vutils.save_image(tensor=tensor_minibatch, filename=image_path+file_name, normalize=True)
