"""
Handle Simple ITK image resgistration.

sitk_multi_spect_registration.py
Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
All rights reserved.
Code revision 2021, Sam Williams, Sentera Inc.
"""

import math
import time

import cv2
import numpy as np
import SimpleITK as Sitk

from imgreg import multi_spect_common, multi_spect_reg_config


class AlignmentResultsT:
    """Alignment results."""

    def __init__(self):
        """Initialize alignment results."""
        self.opt_stop_cond = {}
        self.opt_stop_it = {}
        self.opt_stop_metric_val = {}
        self.init_metric_val = {}
        self.final_metric_val = {}
        self.alignment_transform = {}
        self.corner_points = {}
        self.metric_logs = {}
        self.successful = {}


# this class implements multispectral registration using the Simple ITK Library
class SitkRegistration:
    """Simple ITK registration."""

    # The constructor takes a string parameter path to a configuration file
    def __init__(
        self,
        config_file_name,
        input_dataset_path=None,
        output_dataset_path=None,
        failure_dataset_path=None,
    ):
        """Initialize Simple ITK registration."""
        self.config_file_name = config_file_name
        self.config = multi_spect_reg_config.RegConfigT(
            self.config_file_name,
            input_dataset_path,
            output_dataset_path,
            failure_dataset_path,
        )
        self.metric_vals = {}

    def align(
        self,
        multi_ch_image,
        init_transforms=None,
        print_output=False,
        skip_metric_evaluate=True,
    ):
        """Perform alignment on all channels."""
        # check the inputs are of the right type
        assert isinstance(multi_ch_image, np.ndarray)
        # check the datatypes of the inputs are float32
        assert multi_ch_image.dtype == np.float32
        # check that multi_ch_image has a 3rd dimension
        assert len(multi_ch_image.shape) > 2
        # get the number of channels
        num_channels = multi_ch_image.shape[2]
        # check the number of channels matches the camera parameters
        assert num_channels == len(self.config.ordered_channel_names)
        # if some initial transform is passed, check that it's the right length
        if init_transforms is not None:
            assert type(init_transforms) == dict
            assert len(init_transforms.keys()) == num_channels - 1

        # A python list of the original images
        original_numpy_images = []
        # A dictionary mapping channel names to image indicies
        channel_indexes = {}
        # For each channel (in order)
        for i, ch_name in enumerate(self.config.ordered_channel_names):
            # get the original frame from the right index
            original_numpy_images.append(multi_ch_image[:, :, i])
            # save the channel's index
            channel_indexes[ch_name] = i

        # get the fixed channel's index
        fixed_ch_idx = channel_indexes[self.config.fixed_channel_name]
        # get the shape of the original frame
        original_frame_shape = multi_ch_image[:, :, fixed_ch_idx].shape

        # undistort all image images
        undistorted_img_list = multi_spect_common.undistort_images(
            original_numpy_images,
            self.config.camera_params,
            self.config.ordered_channel_names,
        )

        # convert numpy arrays to ITK images
        itk_img_list = multi_spect_common.convert_to_itk(undistorted_img_list)

        # normalize the input images
        norm_itk_img_list = multi_spect_common.normalize_itk_img(itk_img_list)

        # create the output image
        w = original_frame_shape[0]
        h = original_frame_shape[1]
        aligned_multi_ch_image = np.zeros(shape=(w, h, num_channels), dtype=np.float32)
        # save the fixed frame in the correct index
        aligned_multi_ch_image[:, :, fixed_ch_idx] = original_numpy_images[fixed_ch_idx]

        # reset the previous transformation lookup
        self.alignment_results = AlignmentResultsT()

        # get the fixed frames
        fixed_proc = norm_itk_img_list[fixed_ch_idx]
        fixed_orig = itk_img_list[fixed_ch_idx]

        # setup the corner pixels (for extrema transforms)
        corner_pix = [
            (0, 0),
            (fixed_orig.GetWidth(), 0),
            (fixed_orig.GetWidth(), fixed_orig.GetHeight()),
            (0, fixed_orig.GetHeight()),
        ]
        # create the results object
        self.alignment_results = AlignmentResultsT()
        if print_output:
            print(
                "Fixed channel : ",
                self.config.fixed_channel_name,
                ", at index : ",
                fixed_ch_idx,
            )
        # loop through the moving channels
        for ch_name in self.config.moving_channel_names:
            # lookup the channel index
            ch_idx = channel_indexes[ch_name]
            if print_output:
                print("Processing channel : ", ch_name, ", at index : ", ch_idx)

            # get the config parameters for this channel
            channel_params = self.config.param_map[ch_name]

            # setup the initial transform
            if init_transforms is not None:
                init_tf = init_transforms[ch_name]
            else:
                init_tf = Sitk.AffineTransform(2)

            # get the processing (normalized) and original images for this channel
            moving_proc = norm_itk_img_list[ch_idx]
            moving_orig = itk_img_list[ch_idx]

            # create a channel results object
            ch_results = None
            max_attempts = channel_params["max_alignment_attempts"]
            start_time = time.time()
            success = False
            # Loop over the maximum number of attempts
            for attempt in range(max_attempts):
                self.metric_vals = []
                try:
                    # run alignment
                    aligned_image, ch_results = self.align_channel(
                        fixed_proc,
                        moving_proc,
                        fixed_orig,
                        moving_orig,
                        init_tf,
                        channel_params,
                        original_frame_shape,
                        attempt + 15000,
                        skip_metric_evaluate=skip_metric_evaluate,
                    )
                    # if no exception has been raised, the alignment succeeded
                    success = True
                    # break the "attempt" for loop
                    break
                except RuntimeError:
                    if print_output:
                        print(
                            "Failed Channel Alignment Attempt #",
                            str(attempt + 1),
                            " of ",
                            str(max_attempts),
                        )
            # if success hasn't been set to True, the alignment failed
            if success:
                self.alignment_results.opt_stop_cond[ch_name] = ch_results[0]
                self.alignment_results.opt_stop_it[ch_name] = ch_results[1]
                self.alignment_results.opt_stop_metric_val[ch_name] = ch_results[2]
                self.alignment_results.alignment_transform[ch_name] = ch_results[3]
                self.alignment_results.init_metric_val[ch_name] = ch_results[4]
                self.alignment_results.final_metric_val[ch_name] = ch_results[5]
                self.alignment_results.metric_logs[ch_name] = self.metric_vals
                self.alignment_results.successful[ch_name] = (
                    self.alignment_results.opt_stop_metric_val[ch_name]
                    < channel_params["opt_final_metric_min"]
                )
                # lookup the transformed bounds
                inv_alignment = self.alignment_results.alignment_transform[
                    ch_name
                ].GetInverse()
                corner_points_transformed = [
                    inv_alignment.TransformPoint(pnt) for pnt in corner_pix
                ]
                self.alignment_results.corner_points[
                    ch_name
                ] = corner_points_transformed
                # save the aligned image
                aligned_multi_ch_image[:, :, ch_idx] = Sitk.GetArrayFromImage(
                    aligned_image
                )
            else:
                self.alignment_results.opt_stop_cond[ch_name] = "FAILED"
                self.alignment_results.opt_stop_it[ch_name] = -1
                self.alignment_results.opt_stop_metric_val[ch_name] = -1
                self.alignment_results.alignment_transform[
                    ch_name
                ] = init_tf = Sitk.AffineTransform(2)
                self.alignment_results.init_metric_val[ch_name] = -1
                self.alignment_results.final_metric_val[ch_name] = -1
                self.alignment_results.successful[ch_name] = False
                self.alignment_results.metric_logs[ch_name] = self.metric_vals
                # save the original image
                aligned_multi_ch_image[:, :, ch_idx] = Sitk.GetArrayFromImage(
                    moving_orig
                )
            # print to console
            if print_output:
                print("Channel %s Alignment Complete" % (ch_name))
                print(
                    "\tOptimizer Stop Cond : ",
                    self.alignment_results.opt_stop_cond[ch_name],
                )
                print(
                    "\tOptimizer Stop Iter : ",
                    self.alignment_results.opt_stop_it[ch_name],
                )
                print(
                    "\tOptimizer Metric Val : ",
                    self.alignment_results.opt_stop_metric_val[ch_name],
                )
                print("Channel Align took : ", time.time() - start_time)
                print("-------------------------------------------------")

        # check that the output matches the input dimensions and datatype
        assert multi_ch_image.shape == aligned_multi_ch_image.shape
        assert multi_ch_image.dtype == aligned_multi_ch_image.dtype

        # if enabled, remove all rows and columns that have a zero in at least one channel for every pixel
        if self.config.remove_partial_edges:
            any_zeroes = np.any(aligned_multi_ch_image == 0, axis=2)
            aligned_multi_ch_image = aligned_multi_ch_image[~np.all(any_zeroes, axis=1)]
            aligned_multi_ch_image = aligned_multi_ch_image[
                :, ~np.all(any_zeroes, axis=0)
            ]

        return aligned_multi_ch_image, self.alignment_results

    def align_channel(
        self,
        fixed_op,
        moving_op,
        fixed_of,
        moving_of,
        init_tf,
        channel_params,
        img_shape,
        seed,
        skip_metric_evaluate=True,
    ):
        """Perform alignment on a single channel."""
        reg_method = Sitk.ImageRegistrationMethod()
        self.reg_method = reg_method
        # configures the "space" used for transformations
        reg_method.SetVirtualDomainFromImage(fixed_of)

        # set a mask to only evaluate the metric in the center of the images
        mmb = channel_params["metric_mask_border_size"]
        if mmb > 0:
            metric_mask = np.zeros(shape=img_shape)
            metric_mask[mmb:-mmb, mmb:-mmb] = 1.0
            # apply the mask to both the fixed and the moving images
            reg_method.SetMetricFixedMask(Sitk.GetImageFromArray(metric_mask))
            reg_method.SetMetricMovingMask(Sitk.GetImageFromArray(metric_mask))
        # configure registration from configuration parameters
        reg_method.SetMetricSamplingPercentagePerLevel(
            channel_params["metric_sampling_rate_per_level"]
        )
        reg_method.SetMetricSamplingStrategy(Sitk.ImageRegistrationMethod.REGULAR)
        reg_method.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=channel_params["metric_num_hist_bins"]
        )

        reg_method.SetShrinkFactorsPerLevel(
            shrinkFactors=channel_params["opt_shrink_factors"]
        )
        reg_method.SetSmoothingSigmasPerLevel(
            smoothingSigmas=channel_params["opt_scale_sigmas"]
        )

        reg_method.SetOptimizerAsOnePlusOneEvolutionary(
            numberOfIterations=channel_params["evol_iterations"],
            epsilon=channel_params["evol_epsilon"],
            initialRadius=channel_params["evol_init_rad"],
            growthFactor=channel_params["evol_growth_fact"],
            shrinkFactor=channel_params["evol_shrink_fact"],
            seed=seed,
        )

        reg_method.SetInterpolator(Sitk.sitkLinear)
        reg_method.AddCommand(Sitk.sitkIterationEvent, self._multi_res_update)

        # if no initial transform is given, set it to an identity affine
        if init_tf is None:
            init_tf = Sitk.AffineTransform(2)
        # inPlace doesn't have to be false if reg_method is re-constructed for each channel
        reg_method.SetInitialTransform(init_tf)
        reg_method.SetOptimizerScalesFromPhysicalShift()
        # reg_method.SetOptimizerScales([1e-5,1e-5,1e-5,1e-5,0.5,0.5])

        # Calulate the initial registration metric (before registration)
        if skip_metric_evaluate:
            initial_metric = None
        else:
            initial_metric = reg_method.MetricEvaluate(fixed_of, moving_of)

        # run registration
        alignment_transform = reg_method.Execute(fixed_op, moving_op)

        # transform the input image with the derived transform
        resampler = Sitk.ResampleImageFilter()
        resampler.SetInterpolator(Sitk.sitkLinear)
        resampler.SetReferenceImage(fixed_of)
        resampler.SetTransform(alignment_transform)
        xform_img = resampler.Execute(moving_of)

        # Re-calculate the registration metric (after registration)
        if skip_metric_evaluate:
            final_metric = None
        else:
            final_metric = reg_method.MetricEvaluate(fixed_of, xform_img)

        self.reg_method = None
        return xform_img, (
            reg_method.GetOptimizerStopConditionDescription(),
            reg_method.GetOptimizerIteration(),
            reg_method.GetMetricValue(),
            alignment_transform,
            initial_metric,
            final_metric,
        )

    # optimizer iteration callback
    def _multi_res_update(self):
        if self.reg_method is not None:
            opt_it = self.reg_method.GetOptimizerIteration()
            opt_pos = self.reg_method.GetOptimizerPosition()
            learn_rate = self.reg_method.GetOptimizerLearningRate()
            conv_val = self.reg_method.GetOptimizerConvergenceValue()
            metric = self.reg_method.GetMetricValue()
            self.metric_vals.append((opt_it, opt_pos, learn_rate, conv_val, metric))

    def process_6x_rgb(self, image):
        """Remove edges of RGB channel based on alignment results."""
        # undistort image
        image = cv2.undistort(
            image,
            np.array([[714.0, 0.0, 714.0], [0.0, 952.0, 952.0], [0.0, 0.0, 1.0]]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            None,
            np.array([[714.0, 0.0, 714.0], [0.0, 952.0, 952.0], [0.0, 0.0, 1.0]]),
        )

        left_bounds = {}
        top_bounds = {}
        right_bounds = {}
        bottom_bounds = {}
        # find average location for each boundary and scale up to RGB resolution
        ratio = (5184.0 / 1904.0) / 2.0
        for ch, points in self.alignment_results.corner_points.items():
            left_bounds[ch] = math.ceil((points[0][0] + points[3][0]) * ratio)
            top_bounds[ch] = math.ceil((points[0][1] + points[1][1]) * ratio)
            right_bounds[ch] = math.floor((points[1][0] + points[2][0]) * ratio)
            bottom_bounds[ch] = math.floor((points[2][1] + points[3][1]) * ratio)
        if self.config.remove_partial_edges:
            # find innermost boundaries across all channels
            max_left = max(max(left_bounds.values()), 0)
            max_top = max(max(top_bounds.values()), 0)
            min_right = min(min(right_bounds.values()), 5184)
            min_bottom = min(min(bottom_bounds.values()), 3888)
            # crop image to these boundaries
            image = image[
                max_top
                - top_bounds[self.config.rgb_6x] : min_bottom
                - bottom_bounds[self.config.rgb_6x]
                or None,
                max_left
                - left_bounds[self.config.rgb_6x] : min_right
                - right_bounds[self.config.rgb_6x]
                or None,
                :,
            ]
        else:
            # crop image to boundaries of fixed channel
            image = image[
                max(-top_bounds[self.config.rgb_6x], 0) : min(
                    3888 - bottom_bounds[self.config.rgb_6x], 0
                )
                or None,
                max(-left_bounds[self.config.rgb_6x], 0) : min(
                    5184 - right_bounds[self.config.rgb_6x], 0
                )
                or None,
                :,
            ]
            # fill zeroes to align all corners with those of fixed channel
            image = np.pad(
                image,
                (
                    (
                        max(top_bounds[self.config.rgb_6x], 0),
                        max(3888 - bottom_bounds[self.config.rgb_6x], 0),
                    ),
                    (
                        max(left_bounds[self.config.rgb_6x], 0),
                        max(5184 - right_bounds[self.config.rgb_6x], 0),
                    ),
                    (0, 0),
                ),
            )
        return image
