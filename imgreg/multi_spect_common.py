"""
Common utilities for Simple ITK image registration.

multi_spect_common.py
Copyright (c) 2020, Kostas Alexis, Frank Mascarich, Autonomous Robots Lab, University of Nevada, Reno.
All rights reserved.
"""

import os

import cv2
import numpy as np
import SimpleITK as Sitk


def undistort_images(in_img_list, camera_param_list, ch_names):
    """Undistort image using camera parameters."""
    undistorted_img_list = []
    for i in range(len(in_img_list)):
        camera_params = camera_param_list[ch_names[i]]
        undistorted = cv2.undistort(
            in_img_list[i],
            camera_params.camera_matrix,
            camera_params.dist_vect,
            None,
            camera_params.camera_matrix,
        )
        undistorted_img_list.append(undistorted)
    return undistorted_img_list


def convert_to_itk(in_img_list):
    """Convert numpy array formatted image to Simple ITK."""
    itk_img_list = []
    for in_img in in_img_list:
        itk_img_list.append(Sitk.GetImageFromArray(in_img))
    return itk_img_list


def normalize_itk_img(in_img_list):
    """Perform Simple ITK normalization on an image."""
    norm_img_list = []
    for in_img in in_img_list:
        norm_img_list.append(Sitk.Normalize(in_img))
    return norm_img_list


def get_dataset_paths(ch_paths, file_extention_str=None, num_ch=None):
    """Get paths to all images to be processed."""
    assert type(ch_paths) == list
    if file_extention_str is not None:
        assert type(file_extention_str) == str
    if num_ch is not None:
        assert type(num_ch) == int
    # create the dictionary which maps image ID's to lists of image paths, in the same order as the given paths
    img_dict = {}
    # loop through the channel paths
    for ch_path in ch_paths:
        # make sure the path is a string
        assert type(ch_path) == str
        # loop through all the files in the listed directory
        for img_file in os.listdir(ch_path):
            # if the user passed an extension to match
            if file_extention_str is not None:
                # if the image file does not end in the correct extention
                if not img_file.endswith(file_extention_str):
                    # ignore the file
                    continue
            # get the image ID as the number between the first two underscores
            split = img_file.split("_")
            # convert the string to an integer
            img_id = int(split[1])
            # build the full image path
            img_path = os.path.join(ch_path, img_file)
            # make sure the image path exists
            assert os.path.exists(img_path)
            # if this is the first image with this ID
            if img_id not in img_dict.keys():
                # create a new list at the new ID
                img_dict[img_id] = []
            # add the image to the list at the extracted key
            img_dict[img_id].append(img_path)
    # if the user passed in the number of channels
    if num_ch is not None:
        ids_to_remove = []
        # check each image id
        for img_id in img_dict:
            if len(img_dict[img_id]) != num_ch:
                ids_to_remove.append(img_id)
        for img_id in ids_to_remove:
            del img_dict[img_id]
    # return the dataset paths
    return img_dict


def normalize_channel(channel):
    """Normalize a single channel to 0-1 range."""
    return np.array(channel) / np.max(channel)


def normalize_float_img(img):
    """Normalize each channel in an image to 0-1 range."""
    out_image = np.zeros(shape=img.shape)
    if len(out_image.shape) > 2:
        for i in range(out_image.shape[2]):
            out_image[:, :, i] = normalize_channel(img[:, :, i])
    else:
        out_image = normalize_channel(img)
    return out_image
