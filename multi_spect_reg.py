"""
Perform multi-spectral image registration.

multi_spect_reg.py
Copyright (c) 2020, Kostas Alexis, Frank Mascarich, University of Nevada, Reno.
All rights reserved.
"""

from imgreg import multi_spect_dataset_handling

dataset_handler = multi_spect_dataset_handling.DataSetHandler("cfg/reg_config.ini")
dataset_handler.process_all_images(use_init_transform=True, update_from_previous=True)
