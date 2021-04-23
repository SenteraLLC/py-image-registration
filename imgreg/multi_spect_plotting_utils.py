# multi_spect_plotting_utils.py
# Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
# All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def show_multi_ch_image(img, ch_titles=None):
	# display image channels
	plt.figure(figsize=(16, 12))
	n_ch = img.shape[2]
	ax1=plt.subplot(1, n_ch, 1)
	ax1.imshow(img[:,:,0])
	ax1.title.set_text(ch_titles[0])
	ax1.xaxis.tick_top()
	for i in range(1, n_ch):
		ax=plt.subplot(1, n_ch, i+1)
		ax.imshow(img[:,:,i])
		ax.title.set_text(ch_titles[i])
		ax.xaxis.tick_top()
	plt.tight_layout()
	plt.show()

def normalize_channel(channel):
	return channel / np.max(channel)

def normalize_float_img(img):
	out_image = np.zeros(shape=img.shape)
	if len(img.shape) > 2:
		for i in range(img.shape[2]):
			out_image[:,:,i] = normalize_channel(img[:,:,i])
	else:
		out_image = normalize_channel(img)
	return out_image

def plot_metric_vals(metrics, colors, names, met_idx=4):
	for i,name in enumerate(names):
		vals = []
		for j in range(len(metrics[name])):
			vals.append(metrics[name][j][met_idx])
		plt.plot(vals,color=colors[i],label=name)
	plt.legend()
	plt.show()

def show_alignment_result(in_img, out_img, channel_names, dpi=180):
	assert in_img.shape == out_img.shape
	n_ch = in_img.shape[2]
	plt.figure(figsize=(16, 24), dpi=dpi)
	in_img = normalize_float_img(in_img)
	out_img = normalize_float_img(out_img)
	for i in range(0, (n_ch-1)*2, 2):
		ch_i = int(i / 2) + 1
		ax1 = plt.subplot(n_ch, 2, i+1)
		im1 = ax1.imshow(in_img[:,:,0], cmap=plt.cm.spring, alpha=.75, interpolation='nearest')
		im2 = ax1.imshow(in_img[:,:,ch_i], cmap=plt.cm.spring, alpha=.75, interpolation='nearest')
		ax1.title.set_text('Input Image Alignment : ' + str(channel_names[0]) + " -->" + str(channel_names[ch_i]))

		ax2 = plt.subplot(n_ch, 2, i+2)
		im3 = ax2.imshow(out_img[:,:,0], cmap=plt.cm.spring, alpha=.75, interpolation='nearest')
		im4 = ax2.imshow(out_img[:,:,ch_i], cmap=plt.cm.spring, alpha=.75, interpolation='nearest')
		ax2.title.set_text('Output Image Alignment : ' + str(channel_names[0]) + " -->" + str(channel_names[ch_i]))
	plt.tight_layout() 
	plt.show()

def show_merged(image, channel_list, blend_ch=-1, image_bounds=None, colors=None):
	if image_bounds is not None:
		assert colors is not None
	plt.figure(figsize=(10, 8), dpi=80)
	save_img = normalize_float_img(image)
	# scale to 255
	save_img = save_img * 255.0
	h = image.shape[0]
	w = image.shape[1]
	out_img = np.zeros(shape=(h,w,3))
	for i in range(3):
		if blend_ch != -1:
			out_img[:,:,i] = (save_img[:,:,blend_ch]*0.5) + (save_img[:,:,channel_list[i]]*0.5)
		else:
			out_img[:,:,i] = save_img[:,:,channel_list[i]]
	out_img = out_img.astype(np.uint8)
	ax1 = plt.subplot(1, 1, 1)
	ax1.imshow(out_img)
	if image_bounds is not None:
		frame_boxes = []
		for ch in image_bounds:
			frame_box = Polygon(image_bounds[ch], closed=True, fill=False,linewidth=3,edgecolor=colors[ch])
			frame_boxes.append(frame_box)
		# Create patch collection with specified colour/alpha
		pc = PatchCollection(frame_boxes, match_original=True)
		# Add collection to axes
		ax1.add_collection(pc)
	plt.tight_layout()
	plt.show()
