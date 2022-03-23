"""
Image input/output handling.

multi_spect_image_io.py
Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
All rights reserved.
Code revision 2021, Sam Williams, Sentera Inc.
"""

import logging
import subprocess

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def save_tif_image(image, file_paths):
    """Save single channel tif images."""
    h = image.shape[0]
    w = image.shape[1]
    logger.info(f"Saving tiff image of shape {image.shape}")
    for i in range(len(file_paths)):
        out_img = np.zeros(shape=(h, w, 1), dtype=np.float32)
        out_img[:, :, 0] = image[:, :, i]
        cv2.imwrite(file_paths[i], out_img)


def save_jpg_image(image, path, channel_list, blend_ch=-1):
    """Save jpg image."""
    h = image.shape[0]
    w = image.shape[1]
    logger.info(f"Saving jpg image of shape {image.shape}")
    out_img = np.zeros(shape=(h, w, 3))
    for i in range(3):
        if blend_ch != -1:
            out_img[:, :, i] = (image[:, :, blend_ch] * 0.5) + (
                image[:, :, channel_list[i]] * 0.5
            )
        else:
            out_img[:, :, i] = image[:, :, channel_list[i]]
    out_img = out_img.astype(np.uint8)
    cv2.imwrite(path + ".jpg", out_img)


# creates a numpy array from a list of image paths
def load_image_from_path_list(img_paths, config):
    """Load single channel image."""
    out_image = None
    for i, path in enumerate(img_paths):
        img = np.array(Image.open(path))
        if len(img.shape) > 2:
            img = img[:, :, 0]
        # if its the first channel
        if out_image is None:
            h = img.shape[0]
            w = img.shape[1]
            c = len(img_paths)
            out_image = np.zeros(shape=(h, w, c), dtype=np.float32)
        if config.rgb_6x is not None and i == config.ordered_channel_names.index(
            config.rgb_6x
        ):
            out_image[:, :, i] = np.float32(
                cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            )
        else:
            out_image[:, :, i] = np.float32(img)
    return out_image


# simple loading of 3 channel image with reversed channels
def load_bgr_image(path):
    """Load reverse-channel-order 3 channel image."""
    img = np.array(Image.open(path))
    ch1 = img[:, :, 0].copy()
    img[:, :, 0] = img[:, :, 2]
    img[:, :, 2] = ch1
    return np.array(img)


def _convert_to_degrees(value):
    """
    Convert the GPS coordinates stored in the EXIF to degress in float format.

    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)

    return d + (m / 60.0) + (s / 3600.0)


def copy_exif(source_path, dest_path, exiftool_path, fixed_channel_exif_data=None):
    """Copy metadata and set lat/lon if necessary."""
    command = [
        exiftool_path,
        "-config",
        "cfg/exiftool.cfg",
        "-overwrite_original",
        "-TagsFromFile",
        source_path,
        "-all",
    ]

    if fixed_channel_exif_data is not None:
        latitude = _convert_to_degrees(fixed_channel_exif_data["GPS GPSLatitude"])
        longitude = _convert_to_degrees(fixed_channel_exif_data["GPS GPSLongitude"])
        logger.info(f"Setting coordinates to {latitude}/{longitude}")
        command += [
            f"-gpslatitude={latitude}",
            f"-gpslongitude={longitude}",
        ]
    command.append(dest_path)

    results = subprocess.run(command, capture_output=True)
    if results.returncode != 0:
        raise ValueError("Exiftool command did not run successfully.")
