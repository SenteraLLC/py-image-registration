# multi_spect_image_io.py
# Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
# All rights reserved.

import cv2
import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

def save_tif_image(image, output_path, filename, channel_names):
	h = image.shape[0]
	w = image.shape[1]
	logger.info(f"Saving tiff image of shape {image.shape}")
	for i in range(len(channel_names)):
		out_img = np.zeros(shape=(h,w,1))
		out_img[:,:,0] = image[:,:,i]
		cv2.imwrite(os.path.join(output_path, channel_names[i], filename) + ".tif", out_img)


def save_jpg_image(image, path, channel_list, blend_ch=-1):
	h = image.shape[0]
	w = image.shape[1]
	logger.info(f"Saving jpg image of shape {image.shape}")
	out_img = np.zeros(shape=(h,w,3))
	for i in range(3):
		if blend_ch != -1:
			out_img[:,:,i] = (image[:,:,blend_ch]*0.5) + (image[:,:,channel_list[i]]*0.5)
		else:
			out_img[:,:,i] = image[:,:,channel_list[i]]
	out_img = out_img.astype(np.uint8)
	cv2.imwrite(path + ".jpg", out_img)


# creates a numpy array from a list of image paths 
def load_image_from_path_list(img_paths, config):
	out_image = None
	for i,path in enumerate(img_paths):
		img = np.array(Image.open(path))
		if len(img.shape) > 2:
			img = img[:,:,0]
		# if its the first channel
		if out_image is None:
			h = img.shape[0]
			w = img.shape[1]
			c = len(img_paths)
			out_image = np.zeros(shape=(h, w, c), dtype=np.float32)
		if config.rgb_6x is not None and i == config.ordered_channel_names.index(config.rgb_6x):
			out_image[:,:,i] = np.float32(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
		else:
			out_image[:,:,i] = np.float32(img)
	return out_image

# simple loading of 3 channel image with reversed channels
def load_image(path):
	img = np.array(Image.open(path))
	ch1 = img[:,:,0].copy()
	img[:,:,0] = img[:,:,2]
	img[:,:,2] = ch1
	return np.array(img)