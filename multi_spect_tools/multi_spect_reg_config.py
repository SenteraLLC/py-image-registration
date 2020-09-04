# multi_spect_reg_config.py
# Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
# All rights reserved.

import SimpleITK as sitk
import os
import configparser
import numpy as np

class camera_parameter_t():
	def __init__(self, camera_name, fx, fy, cx, cy, dist_vect):
		self.camera_name = camera_name
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.dist_vect = np.array(dist_vect)
		self.camera_matrix = np.array([ [ fx,   0.0,   cx],
										[0.0,   fy,    cy],
										[0.0,   0.0,  1.0]])

class reg_config_t:
	def __init__(self, config_file_path):
		configDict = configparser.ConfigParser()
		# read the config file
		configDict.read(config_file_path)
		# get the name of the fixed channel
		self.fixed_channel_name				= self.clean_string(configDict['REGISTRATION']['fixed_channel'])
		self.ordered_channel_names			= self.parse_list(configDict['REGISTRATION']['input_channel_order'], str)
		# check that the fixed channel name is also in the ordered channel names
		assert self.fixed_channel_name in self.ordered_channel_names
		# make a list of the moving channel names (all channels except the fixed channel)
		self.moving_channel_names = list(self.ordered_channel_names)
		self.moving_channel_names.remove(self.fixed_channel_name)
		
		# make a dictionary of the per-channel settings
		self.channel_param_map_settings = {}
		for moving_ch_name in self.moving_channel_names:
			self.channel_param_map_settings[moving_ch_name] = str(configDict['REGISTRATION']["param_map_" + moving_ch_name])

		# dataset processing config
		self.channel_paths = {}
		# if the config file specifies settings for dataset processing
		if 'PATHS' in configDict.keys():
			self.input_dataset_path = str(configDict['PATHS']["INPUT_DATASET_PATH"])
			self.output_dataset_path = str(configDict['PATHS']["OUTPUT_DATASET_PATH"])
			self.failure_dataset_path = str(configDict['PATHS']["OUTPUT_FAILURE_PATH"])
			# for each channel build the absolute path
			for name in self.ordered_channel_names:
				sub_dir_path = str(configDict['PATHS'][name+"_SUBDIR"])
				sub_dir_path = os.path.join(self.input_dataset_path, sub_dir_path)
				self.channel_paths[name] = sub_dir_path
			# create a dictionary which maps image IDs to a dictionary which maps channel names to the path of the image for that channel
			self.img_path_dict = {}
			self.load_image_dict(self.channel_paths)

			# get the list of image IDs
			self.image_ids = list(self.img_path_dict.keys())
			self.image_ids.sort()

		# Load Per-Channel Registration Settings
		self.param_map = {}
		for ch_name in self.channel_param_map_settings:
			p_map = self.channel_param_map_settings[ch_name]
			self.param_map[ch_name] = {}
			self.param_map[ch_name]['max_alignment_attempts']	= int(configDict[p_map]["max_alignment_attempts"])
			self.param_map[ch_name]['metric_num_hist_bins']	= int(configDict[p_map]["metric_num_hist_bins"])
			self.param_map[ch_name]['metric_mask_border_size']	= int(configDict[p_map]["metric_mask_border_size"])
			self.param_map[ch_name]['metric_sampling_rate_per_level']	= self.parse_list(configDict[p_map]["metric_sampling_rate_per_level"], float)

			self.param_map[ch_name]['opt_shrink_factors']	= self.parse_list(configDict[p_map]["opt_shrink_factors"], int)
			self.param_map[ch_name]['opt_scale_sigmas']		= self.parse_list(configDict[p_map]["opt_scale_sigmas"], float)
			self.param_map[ch_name]['opt_final_metric_min']	= float(configDict[p_map]["opt_final_metric_min"])
			
			self.param_map[ch_name]['evol_epsilon']		= float(configDict[p_map]["evol_epsilon"])
			self.param_map[ch_name]['evol_iterations']	= int(configDict[p_map]["evol_iterations"])
			self.param_map[ch_name]['evol_init_rad']	= float(configDict[p_map]["evol_init_rad"])
			self.param_map[ch_name]['evol_growth_fact']	= float(configDict[p_map]["evol_growth_fact"])
			self.param_map[ch_name]['evol_shrink_fact']	= float(configDict[p_map]["evol_shrink_fact"])

		# Load Per-Channel Camera Settings
		self.camera_params = {}
		for ch_name in self.ordered_channel_names:
			self.load_cam_config(configDict, ch_name)

	def load_image_dict(self, data_set_paths_dict):
		self.img_path_dict = {}
		for ch_name in self.ordered_channel_names:
			file_list = os.listdir(data_set_paths_dict[ch_name])
			# print("Ch %s found %i files"%(ch_name, len(file_list)))
			for file_name in file_list:
				img_id = int(list(file_name.split('_'))[1])
				#if img_id == 1:
				#	 print("Found ID 1 for channel : ", ch_name)
				if img_id not in self.img_path_dict.keys():
					self.img_path_dict[img_id] = {}
				self.img_path_dict[img_id][ch_name] = os.path.join(data_set_paths_dict[ch_name], file_name)
		# remove any images which don't have images for all channels
		bad_image_ids = []
		for img_id in self.img_path_dict:
			if len(self.img_path_dict[img_id].keys()) != len(self.ordered_channel_names):
				# print("Bad ID : %i, number of keys: %i, number of channels: %i"%(img_id, len(self.img_path_dict[img_id].keys()), len(self.ordered_channel_names)))
				# print(list(self.img_path_dict[img_id].keys()))
				bad_image_ids.append(img_id)

		for img_id in bad_image_ids:
			del self.img_path_dict[img_id]
		
		# get the list of image IDs
		self.image_ids = list(self.img_path_dict.keys())

	def get_img_paths(self, id):
		paths = self.img_path_dict[id]
		path_list = []
		for ch_name in self.ordered_channel_names:
			path_list.append(paths[ch_name])
		return path_list

	def load_cam_config(self, configDict, ch_name):
		cam_name = "CAM_" + str(ch_name)
		try:
			fx					= float(configDict[cam_name]['fx'])
			fy					= float(configDict[cam_name]['fy'])
			cx					= float(configDict[cam_name]['cx'])
			cy					= float(configDict[cam_name]['cy'])
			dist_vect			= self.parse_list(configDict[cam_name]['dist_vect'], float)
			self.camera_params[ch_name] = camera_parameter_t(ch_name, fx, fy, cx, cy, dist_vect)
		except KeyError as e:
			raise Exception("Error While Parsing Camera Configuration : Config file does not contain key : " + str(e))

	def clean_string(self, in_string):
		in_string = in_string.strip()
		in_string = in_string.replace('\t', '')
		in_string = in_string.replace(' ', '')
		return in_string

	def parse_list(self, in_string, d_type):
		in_string = self.clean_string(in_string)
		string_list = in_string.split(',')
		result = []
		for string in string_list:
			result.append(d_type(string))
		return result

