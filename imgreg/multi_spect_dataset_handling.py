# multi_spect_dataset_handling.py
# Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
# All rights reserved.

import os
import time
from imgreg import multi_spect_common
from imgreg import sitk_multi_spect_registration
from imgreg import multi_spect_image_io
class data_set_handler:
	def __init__(self, config_file, input_dataset_path=None, output_dataset_path=None, failure_dataset_path=None):
		self.init_transforms = None
		# Create a registration object
		self.sitk_reg_obj = sitk_multi_spect_registration.sitk_registration(config_file, input_dataset_path, output_dataset_path, failure_dataset_path)
		# Show which image ID's were loaded
		print("Valid Image IDs from dataset: \n", self.sitk_reg_obj.config.image_ids)
		self.output_path = self.sitk_reg_obj.config.output_dataset_path
		self.bad_alignment_output_path = self.sitk_reg_obj.config.failure_dataset_path

	def set_init_transform_from_prev(self, results):
		self.init_transforms = results.alignment_transform

	def all_success(self, table):
		for key in table:
			if not table[key]:
				return False
		return True

	def process_all_images(self, use_init_transform=True, update_from_previous=True):
		# loop through all the loaded image id's
		for img_id in self.sitk_reg_obj.config.image_ids:
			print("Aligning image ID: %i"%img_id)
			# build the output file path
			file_name = "aligned_" + str(img_id) + self.sitk_reg_obj.config.image_extension
			# Run the alignment in a try/catch, any exceptions will be printed but ignored
			try:
				# load the image from the path lookup
				np_image = multi_spect_image_io.load_image_from_path_list(self.sitk_reg_obj.config.get_img_paths(img_id))
				# if we're using initial transforms and it's not None
				if use_init_transform and self.init_transforms is not None:
					init_xform = self.init_transforms
				else:
					init_xform = None
				# perform the alignment
				output_image, results = self.sitk_reg_obj.align(np_image,init_transforms=init_xform, print_output=True)
				print("Alignment Complete")
				# if the optimizer's final metric quality is below the min threshold, and we used the init transform
				if not self.all_success(results.successful) and init_xform is not None:
					# Try to align again, without the initial transform
					print("Poor Quality Optimization Found, Re-aligning without initial transform")
					output_image, results = self.sitk_reg_obj.align(np_image, init_transforms=None,print_output=True)
					# if the alignment failed again
					if not self.all_success(results.successful):
						print("Alignment failed again, saving to bad alignment directory")
						# this is a misaligned image...
						if self.sitk_reg_obj.config.image_extension == ".jpg":
							multi_spect_image_io.save_jpg_image(output_image, os.path.join(self.bad_alignment_output_path, file_name), [2,1,0], 3)
						elif self.sitk_reg_obj.config.image_extension == ".tif":
							multi_spect_image_io.save_tif_image(output_image, self.bad_alignment_output_path, file_name, self.sitk_reg_obj.config.ordered_channel_names)
						continue

				# if the optimizer's final metric quality is above the min threshold
				if self.all_success(results.successful):
					print("Successul Alignment, saving result")
					# this is an aligned image 
					if self.sitk_reg_obj.config.image_extension == ".jpg":
						multi_spect_image_io.save_jpg_image(output_image, os.path.join(self.output_path, file_name), [2,1,0], 3)
					elif self.sitk_reg_obj.config.image_extension == ".tif":
						multi_spect_image_io.save_tif_image(output_image, self.output_path, file_name, self.sitk_reg_obj.config.ordered_channel_names)
					# update the init transform if flag is set
					if update_from_previous:
						print("Updating initial transform from previous result")
						self.set_init_transform_from_prev(results)
				else:
					print("Alignment Failed, Saving to bad alignment directory")
					# this is a misaligned image without an initial transform
					if self.sitk_reg_obj.config.image_extension == ".jpg":
						multi_spect_image_io.save_jpg_image(output_image, os.path.join(self.bad_alignment_output_path, file_name), [2,1,0], 3)
					elif self.sitk_reg_obj.config.image_extension == ".tif":
						multi_spect_image_io.save_tif_image(output_image, self.bad_alignment_output_path, file_name, self.sitk_reg_obj.config.ordered_channel_names)

			except RuntimeError as e:
				print("Runtime Error : ", e)
			except Exception as e:
				print("Exception Occurred : ", e)
				print("Failed to process image: ", img_id)
			except :
				print("Exception Occurred")
				print("Failed to process image: ", img_id)