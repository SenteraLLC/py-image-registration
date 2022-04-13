"""
Handle and process images.

multi_spect_dataset_handling.py
Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
All rights reserved.
Code revision 2021, Sam Williams, Sentera Inc.
"""

import os
import traceback

import cv2
import imgparse
from tqdm import tqdm

from imgreg import multi_spect_image_io, sitk_multi_spect_registration


class DataSetHandler:
    """Handle and process images."""

    def __init__(
        self,
        config_file,
        input_dataset_path=None,
        output_dataset_path=None,
        failure_dataset_path=None,
    ):
        """Initialize dataset handler."""
        self.init_transforms = None
        # Create a registration object
        self.sitk_reg_obj = sitk_multi_spect_registration.SitkRegistration(
            config_file, input_dataset_path, output_dataset_path, failure_dataset_path
        )
        # Show which image ID's were loaded
        print("Valid Image IDs from dataset: \n", self.sitk_reg_obj.config.image_ids)
        self.output_path = self.sitk_reg_obj.config.output_dataset_path
        self.bad_alignment_output_path = self.sitk_reg_obj.config.failure_dataset_path

    def set_init_transform_from_prev(self, results):
        """Update initial transform."""
        self.init_transforms = results.alignment_transform

    def all_success(self, table):
        """Print true if all results in table indicate success."""
        for key in table:
            if not table[key]:
                return False
        return True

    def save_image(self, output_image, file_name, output_path, img_id):
        """Save images in jpg or tif format."""
        img_paths = self.sitk_reg_obj.config.get_img_paths(img_id)
        fixed_path = img_paths[
            self.sitk_reg_obj.config.ordered_channel_names.index(
                self.sitk_reg_obj.config.fixed_channel_name
            )
        ]
        fixed_channel_exif_data = imgparse.get_exif_data(fixed_path)
        if self.sitk_reg_obj.config.image_extension == ".jpg":
            output_file_path = os.path.join(output_path, file_name)
            multi_spect_image_io.save_jpg_image(
                output_image, output_file_path, [2, 1, 0], 3
            )
            multi_spect_image_io.copy_exif(fixed_path, output_file_path, "exiftool")
        elif self.sitk_reg_obj.config.image_extension == ".tif":
            # copy channel paths dict in case it's edited by rgb_6x code
            channel_paths = self.sitk_reg_obj.config.channel_paths.copy()
            if self.sitk_reg_obj.config.rgb_6x is not None:
                rgb_input_path = self.sitk_reg_obj.config.get_img_paths(img_id)[
                    self.sitk_reg_obj.config.ordered_channel_names.index(
                        self.sitk_reg_obj.config.rgb_6x
                    )
                ]
                # load original RGB image
                rgb_image = multi_spect_image_io.load_bgr_image(rgb_input_path)
                # align/crop
                rgb_image = self.sitk_reg_obj.process_6x_rgb(rgb_image)
                # output RGB image as .jpg
                rgb_path = channel_paths.pop(self.sitk_reg_obj.config.rgb_6x)
                rgb_subdir = os.path.split(rgb_path)[-1]
                rgb_file_path = (
                    os.path.join(output_path, rgb_subdir, file_name) + ".jpg"
                )
                cv2.imwrite(rgb_file_path, rgb_image)
                multi_spect_image_io.copy_exif(
                    rgb_input_path, rgb_file_path, "exiftool", fixed_channel_exif_data
                )
            output_file_paths = [
                os.path.join(output_path, os.path.split(p)[-1], file_name) + ".tif"
                for p in channel_paths.values()
            ]
            multi_spect_image_io.save_tif_image(output_image, output_file_paths)
            for i, ofp in enumerate(output_file_paths):
                multi_spect_image_io.copy_exif(
                    img_paths[i], ofp, "exiftool", fixed_channel_exif_data
                )

    def process_all_images(
        self,
        use_init_transform=True,
        update_from_previous=True,
        skip_metric_evaluate=True,
        print_output=True,
    ):
        """Perform registration on all images."""
        # loop through all the loaded image id's
        for img_id in tqdm(self.sitk_reg_obj.config.image_ids, desc="Register"):
            if print_output:
                print("Aligning image ID: %i" % img_id)
            # build the output file path
            file_name = "aligned_" + str(img_id)
            # Run the alignment in a try/catch, any exceptions will be printed but ignored
            try:
                # load the image from the path lookup
                np_image = multi_spect_image_io.load_image_from_path_list(
                    self.sitk_reg_obj.config.get_img_paths(img_id),
                    self.sitk_reg_obj.config,
                )
                # if we're using initial transforms and it's not None
                if use_init_transform and self.init_transforms is not None:
                    init_xform = self.init_transforms
                else:
                    init_xform = None
                # perform the alignment
                output_image, results = self.sitk_reg_obj.align(
                    np_image,
                    init_transforms=init_xform,
                    print_output=print_output,
                    skip_metric_evaluate=skip_metric_evaluate,
                )
                if print_output:
                    print("Alignment Complete")
                # if the optimizer's final metric quality is below the min threshold, and we used the init transform
                if not self.all_success(results.successful) and init_xform is not None:
                    # Try to align again, without the initial transform
                    if print_output:
                        print(
                            "Poor Quality Optimization Found, Re-aligning without initial transform"
                        )
                    output_image, results = self.sitk_reg_obj.align(
                        np_image,
                        init_transforms=None,
                        print_output=print_output,
                        skip_metric_evaluate=skip_metric_evaluate,
                    )
                    # if the alignment failed again
                    if not self.all_success(results.successful):
                        if print_output:
                            print(
                                "Alignment failed again, saving to bad alignment directory"
                            )
                        # this is a misaligned image...
                        self.save_image(
                            output_image,
                            file_name,
                            self.bad_alignment_output_path,
                            img_id,
                        )
                        continue

                # if the optimizer's final metric quality is above the min threshold
                if self.all_success(results.successful):
                    if print_output:
                        print("Successul Alignment, saving result")
                    # this is an aligned image
                    self.save_image(output_image, file_name, self.output_path, img_id)
                    # update the init transform if flag is set
                    if update_from_previous:
                        if print_output:
                            print("Updating initial transform from previous result")
                        self.set_init_transform_from_prev(results)
                else:
                    if print_output:
                        print("Alignment Failed, Saving to bad alignment directory")
                    # this is a misaligned image without an initial transform
                    self.save_image(
                        output_image, file_name, self.bad_alignment_output_path, img_id
                    )

            except RuntimeError as e:
                print(traceback.format_exc())
                print("Runtime Error : ", e)
            except Exception as e:
                print("Exception Occurred : ", e)
                print("Failed to process image: ", img_id)
