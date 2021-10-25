"""
Registration configuration handling.

multi_spect_reg_config.py
Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
All rights reserved.
"""

import configparser
import os

import numpy as np


class CameraParameterT:
    """Camera Parameters class."""

    def __init__(self, camera_name, fx, fy, cx, cy, dist_vect):
        """Initialize camera parameters."""
        self.camera_name = camera_name
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.dist_vect = np.array(dist_vect)
        self.camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


class RegConfigT:
    """Registration configuration class."""

    def __init__(
        self,
        config_file_path,
        input_dataset_path=None,
        output_dataset_path=None,
        failure_dataset_path=None,
    ):
        """Initialize config."""
        config_dict = configparser.ConfigParser()
        # read the config file
        config_dict.read(config_file_path)
        # get the name of the fixed channel
        self.fixed_channel_name = self.clean_string(
            config_dict["REGISTRATION"]["fixed_channel"]
        )
        self.ordered_channel_names = self.parse_list(
            config_dict["REGISTRATION"]["input_channel_order"], str
        )
        # check that the fixed channel name is also in the ordered channel names
        assert self.fixed_channel_name in self.ordered_channel_names
        # make a list of the moving channel names (all channels except the fixed channel)
        self.moving_channel_names = list(self.ordered_channel_names)
        self.moving_channel_names.remove(self.fixed_channel_name)

        # make a dictionary of the per-channel settings
        self.channel_param_map_settings = {}
        for moving_ch_name in self.moving_channel_names:
            self.channel_param_map_settings[moving_ch_name] = str(
                config_dict["REGISTRATION"]["param_map_" + moving_ch_name]
            )

        # dataset processing config
        self.channel_paths = {}
        # if the config file specifies settings for dataset processing
        if "PATHS" in config_dict.keys():
            self.input_dataset_path = input_dataset_path or str(
                config_dict["PATHS"]["INPUT_DATASET_PATH"]
            )
            self.output_dataset_path = output_dataset_path or str(
                config_dict["PATHS"]["OUTPUT_DATASET_PATH"]
            )
            self.failure_dataset_path = failure_dataset_path or str(
                config_dict["PATHS"]["OUTPUT_FAILURE_PATH"]
            )

            # for each channel build the absolute path
            for name in self.ordered_channel_names:
                sub_dir_path = config_dict["PATHS"][name + "_SUBDIR"]
                sub_dir_path = os.path.join(self.input_dataset_path, sub_dir_path)
                self.channel_paths[name] = sub_dir_path
            # create a dictionary which maps image IDs to a dictionary which maps channel names to the path of the image for that channel
            self.img_path_dict = {}
            self.load_image_dict(self.channel_paths)

            # get the list of image IDs
            self.image_ids = list(self.img_path_dict.keys())
            self.image_ids.sort()

            # create output directories if they don't exist
            if not os.path.exists(self.output_dataset_path):
                os.mkdir(self.output_dataset_path)
            if not os.path.exists(self.failure_dataset_path):
                os.mkdir(self.failure_dataset_path)
            if self.image_extension == ".tif":
                for c in self.channel_paths.values():
                    output_path = os.path.join(
                        self.output_dataset_path, os.path.split(c)[-1]
                    )
                    failure_path = os.path.join(
                        self.failure_dataset_path, os.path.split(c)[-1]
                    )
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)
                    if not os.path.exists(failure_path):
                        os.mkdir(failure_path)

        # Load Per-Channel Registration Settings
        self.param_map = {}
        for ch_name in self.channel_param_map_settings:
            p_map = self.channel_param_map_settings[ch_name]
            self.param_map[ch_name] = {}
            self.param_map[ch_name]["max_alignment_attempts"] = int(
                config_dict[p_map]["max_alignment_attempts"]
            )
            self.param_map[ch_name]["metric_num_hist_bins"] = int(
                config_dict[p_map]["metric_num_hist_bins"]
            )
            self.param_map[ch_name]["metric_mask_border_size"] = int(
                config_dict[p_map]["metric_mask_border_size"]
            )
            self.param_map[ch_name]["metric_sampling_rate_per_level"] = self.parse_list(
                config_dict[p_map]["metric_sampling_rate_per_level"], float
            )

            self.param_map[ch_name]["opt_shrink_factors"] = self.parse_list(
                config_dict[p_map]["opt_shrink_factors"], int
            )
            self.param_map[ch_name]["opt_scale_sigmas"] = self.parse_list(
                config_dict[p_map]["opt_scale_sigmas"], float
            )
            self.param_map[ch_name]["opt_final_metric_min"] = float(
                config_dict[p_map]["opt_final_metric_min"]
            )

            self.param_map[ch_name]["evol_epsilon"] = float(
                config_dict[p_map]["evol_epsilon"]
            )
            self.param_map[ch_name]["evol_iterations"] = int(
                config_dict[p_map]["evol_iterations"]
            )
            self.param_map[ch_name]["evol_init_rad"] = float(
                config_dict[p_map]["evol_init_rad"]
            )
            self.param_map[ch_name]["evol_growth_fact"] = float(
                config_dict[p_map]["evol_growth_fact"]
            )
            self.param_map[ch_name]["evol_shrink_fact"] = float(
                config_dict[p_map]["evol_shrink_fact"]
            )

        # Load Per-Channel Camera Settings
        self.camera_params = {}
        for ch_name in self.ordered_channel_names:
            self.load_cam_config(config_dict, ch_name)

        if "OPTIONS" in config_dict.keys():
            if "remove_partial_edges" in config_dict["OPTIONS"].keys():
                self.remove_partial_edges = (
                    config_dict["OPTIONS"]["remove_partial_edges"] == "True"
                )
            else:
                self.remove_partial_edges = False

            if "rgb_6x" in config_dict["OPTIONS"].keys():
                self.rgb_6x = self.clean_string(config_dict["OPTIONS"]["rgb_6x"])
                # check that the rgb_6x channel name is also in the ordered channel names
                assert self.rgb_6x in self.ordered_channel_names
            else:
                self.rgb_6x = None
        else:
            self.remove_partial_edges = False
            self.rgb_6x = None

    def load_image_dict(self, data_set_paths_dict):
        """Load images that appear in all channels."""
        self.img_path_dict = {}
        for ch_name in self.ordered_channel_names:
            file_list = os.listdir(data_set_paths_dict[ch_name])
            # print("Ch %s found %i files"%(ch_name, len(file_list)))
            # identify image type
            if file_list[0].endswith(".jpg"):
                if ch_name == self.fixed_channel_name:
                    self.image_extension = ".jpg"
            elif file_list[0].endswith(".tif") or file_list[0].endswith(".tiff"):
                if ch_name == self.fixed_channel_name:
                    self.image_extension = ".tif"
            else:
                print(
                    f"Image file {file_list[0]} is not of a supported type. (.jpg, ,tif, .tiff)"
                )
                raise TypeError
            for file_name in file_list:
                img_id = int(list(file_name.replace(".", "_").split("_"))[1])
                # if img_id == 1:
                # 	 print("Found ID 1 for channel : ", ch_name)
                if img_id not in self.img_path_dict.keys():
                    self.img_path_dict[img_id] = {}
                self.img_path_dict[img_id][ch_name] = os.path.join(
                    data_set_paths_dict[ch_name], file_name
                )
        # remove any images which don't have images for all channels
        bad_image_ids = []
        for img_id in self.img_path_dict:
            if len(self.img_path_dict[img_id].keys()) != len(
                self.ordered_channel_names
            ):
                # print("Bad ID : %i, number of keys: %i, number of channels: %i"%(img_id, len(self.img_path_dict[img_id].keys()), len(self.ordered_channel_names)))
                # print(list(self.img_path_dict[img_id].keys()))
                bad_image_ids.append(img_id)

        for img_id in bad_image_ids:
            del self.img_path_dict[img_id]

        # get the list of image IDs
        self.image_ids = list(self.img_path_dict.keys())

    def get_img_paths(self, index):
        """Get image paths from config."""
        paths = self.img_path_dict[index]
        path_list = []
        for ch_name in self.ordered_channel_names:
            path_list.append(paths[ch_name])
        return path_list

    def load_cam_config(self, config_dict, ch_name):
        """Load camera configuration from config."""
        cam_name = "CAM_" + str(ch_name)
        try:
            fx = float(config_dict[cam_name]["fx"])
            fy = float(config_dict[cam_name]["fy"])
            cx = float(config_dict[cam_name]["cx"])
            cy = float(config_dict[cam_name]["cy"])
            dist_vect = self.parse_list(config_dict[cam_name]["dist_vect"], float)
            self.camera_params[ch_name] = CameraParameterT(
                ch_name, fx, fy, cx, cy, dist_vect
            )
        except KeyError as e:
            raise Exception(
                "Error While Parsing Camera Configuration : Config file does not contain key : "
                + str(e)
            )

    def clean_string(self, in_string):
        """Remove whitespace from string."""
        in_string = in_string.strip()
        in_string = in_string.replace("\t", "")
        in_string = in_string.replace(" ", "")
        return in_string

    def parse_list(self, in_string, d_type):
        """Parse list from config."""
        in_string = self.clean_string(in_string)
        string_list = in_string.split(",")
        result = []
        for string in string_list:
            result.append(d_type(string))
        return result
