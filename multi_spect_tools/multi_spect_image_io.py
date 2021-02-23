# multi_spect_image_io.py
# Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
# All rights reserved.

import matplotlib.image as plt_image
import cv2
import numpy as np

def save_jpg_image(image, path, channel_list, blend_ch=-1):
	h = image.shape[0]
	w = image.shape[1]
	number_of_channels = len(channel_list)
	print(image.shape)
	out_img = np.zeros(shape=(h,w,number_of_channels))
	for i in range(number_of_channels):
		if blend_ch != -1:
			out_img[:,:,i] = (image[:,:,blend_ch]*0.5) + (image[:,:,channel_list[i]]*0.5)
		else:
			out_img[:,:,i] = image[:,:,channel_list[i]]
	out_img = out_img.astype(np.uint16)
	cv2.imwrite(path, out_img)

# creates a numpy array from a list of image paths 
def load_image_from_path_list(img_paths):
	out_image = None
	for i,path in enumerate(img_paths):
		img = plt_image.imread(path)
		if len(img.shape) > 2:
			img = img[:,:,0]
		# if its the first channel
		if out_image is None:
			h = img.shape[0]
			w = img.shape[1]
			c = len(img_paths)
			out_image = np.zeros(shape=(h, w, c), dtype=np.float32)
		out_image[:,:,i] = np.float32(img)
	return out_image