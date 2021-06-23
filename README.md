# Multi Spectral Registration

This package implements multi-channel image registration.

This package is written for Python 3, and has been tested on Python 3.5 and Python 3.8.

Registration is the process of finding an optimal transformation between a "fixed" and a "moving" image. This package allows the user to select a single fixed channel, to which any number of additional channels will be transformed. 

The exposed API consists of a single object _sitk_registration_ which loads a configuration file in which the user specifies registration parameters. The object also exposes a single function, _align_ which accepts a multi-channel image as input, and returns the aligned image alongside an _alignment_results_t_ object. The aligned image will be in the same channel order as the input image, as well as of the same dimension and depth. 

The snippet below demonstrates the simplest possible usage while assuming that the _input_image_ object is loaded elsewhere:
 ```python
    # import the multispectral registration package
    from multi_spect_tools import sitk_multi_spect_registration 
    # Create a registration object
    sitk_reg_obj = sitk_multi_spect_registration.sitk_registration("cfg/reg_config.ini")
    # align the input image
    aligned_image, results = sitk_reg_object.align(input_image)
 ```

## Install Dependencies
1. pip3 install numpy
2. pip3 install opencv-python
3. pip3 install SimpleITK

## Optional Dependencies
4. pip3 install matplotlib

Matplotlib is only used for image file I/O. If the built-in dataset processing is not utilized, this dependency is not required.

## Install with `poetry` on Linux

1) [Set up SSH](https://github.com/SenteraLLC/install-instructions/blob/master/ssh_setup.md)
2) Install [pyenv](https://github.com/SenteraLLC/install-instructions/blob/master/pyenv.md) and [poetry](https://python-poetry.org/docs/#installation)
3) Install package

        git clone git@github.com:SenteraLLC/py-image-registration.git
        cd py-image-registration
        pyenv install $(cat .python-version)
        poetry install
        
4) *Optional* Set up ``pre-commit`` to ensure all commits to adhere to **black** and **PEP8** style conventions.

        poetry run pre-commit install

## Usage

#### Notebooks
This package contains several Jupyter Notebooks to demonstrate API usage.
1. _Sitk_Demo_Registration.ipynb_ demonstrates the lowest level functionality available
for loading datasets, running alignment and visualizing the result. 
2. _Dataset_processing_notebook.ipynb_ mimics _multi_spect_reg.py_, and will load an entire dataset and process it.

#### Configuration
The _cfg/reg_config.ini_ contains all applicable parameters exposed for running alignment.

_.ini_ files are organized by sections:

1. **REGISTRATION**
    1. **input_channel_order** : a comma-separated list of strings containing the name of each channel in the same order as the input image. 
    2. **fixed_channel** : a string giving the name of the fixed channel. This string must also be present in the *input_channel_order*.
    3. **param_maps_<CH_NAME>** : The registration parameter map to associate with each channel. While each moving channel must be associated with a parameter map, multiple channels may share the same parameter map.
    
2. **REG_MAP_\<NAME\>**
    1. **max_alignment_attempts** (_int_) : In some cases, the alignment process will throw an exception when the current solution transforms the moving image off of the fixed image. The registration process will reattempt the alignment with a new seed this many times.
    1. **metric_num_hist_bins** (_int_) : the number of histogram bins to use for Mattes's Mutual Information Metric.
    1. **metric_mask_border_size** (int) : The size of a binary mask border to place around both the moving and fixed images, within which the metric will not be evaluated. This should be set to a number greater than the largest expected translation for any associated channel.
    1. **metric_sampling_rate_per_level** ([_float,..._]) : the proportion of pixels to sample at each optimizer level (0.0 -> 1.0). 
    1. **opt_shrink_factors** ([_int,..._]) : The resolution shrink factors for optimization (shrinking by 2 reduces the imager resolution by 2)
    1. **opt_scale_sigmas** ([_float,..._]): The blur to apply at each optimization resolution
    1. **opt_final_metric_min** (_float_) : Optimizations with final metric values above this value will be considered "failures", and will have their success flags set to False.
    1. **evol_epsilon** (_float_) : The convergence threshold, when the optimizer's convergence metric falls below this epsilon, the optimizer concludes that it has converged. 
    1. **evol_iterations** (_int_): The maximum number of iterations of the optimizer.
    1. **evol_init_rad** (_float_): Initial Radius - the mutation variance is initialized at this value. For channels with larger expected translation values, this value should be enlarged to increase the likelihood the global minima is found. If an initial transformation is given, this value can be substantially reduced for faster convergence.
    1. **evol_growth_fact** (_float_): Evolutionary growth factor - mutation variance grows by this value while the optimizer's metric is increasing ( > 1.0 ). For more difficult channels, this value should be increased. 
    1. **evol_shrink_fact** (_float_): Evolutionary shrink factor - mutation variance shrinks by this value while the optimizer's metric is decreasing ( < 1.0 ).
    
3. **CAM_<CH_NAME>**
    1. **h**: height of image in pixels
    2. **w**: width of image in pixels   
    2. **fx**: focal point X   
    2. **fy**: focal point Y   
    2. **cx**: center X  
    2. **cy**: center Y   
    2. **dist_vect**: Distortion Parameter Vector (_see cv::undistort_) 

4. **PATHS** - Optional parameters for dataset processing.
    1. **INPUT_DATASET_PATH** : local or absolute path to the input image dataset.
    2. **OUTPUT_DATASET_PATH** : local or absolute path to a directory where the aligned images will be saved.
    3. **OUTPUT_FAILURE_PATH** : local or absolute path to a directory where images which failed to align will be saved.
    4. **<CH_NAME>_PATH** : The name of the directory within the **INPUT_DATASET_PATH** containing the channel's images. 
    CH_NAME must match the name given in the **REGISTRATION** parameters above.

### Alignment Results
Alongside the returned image, an object is returned containing the results of the alignment process. The object contains a set of dictionaries in which channel names are used as keys to sets of results for each channel. Specifically, the object contains the following member dictionaries:
1. 	opt_stop_cond 		- a string describing the optimizer's final stopping condition.
2.	opt_stop_it 		- an integer representing the optimizer's final stopping iteration.
3.	opt_stop_metric_val - the optimizer's final metric value upon stopping. This is the metric which is thresholded against the configuration file's parameter _opt_final_metric_min_. 
4.	init_metric_val 	- the channel's metric value before the transformation.
5.	final_metric_val 	- the channel's metric value after the transformation.
6.	alignment_transform - the transformation applied to this channel
7.	corner_points 		- the corner points of the original image transformed by the derived transformation to indicate the current position of the frame in the aligned image.
8.	metric_logs		    - a list of the metric values calculated at each iteration of the optimizer.
9.	successful			- a boolean flag indicating whether or not the alignment was successful. This flag will be set to false if the optimizer's minimum value is not met, or if the maximum number of alignment attempts has been reached. This flag may be true even when the optimizer reached it's maximum number of iterations.


### Dataset File Structure
The _INPUT_DATASET_PATH_ must be a directory with subdirectories each containing the images for the respective channel:
 ```
 - INPUT_DATASET_PATH
    - 0-Blue
        - IMG_0001_Blue.tif
    - 1-Green
    - 2-Red
    - 3-NIR
```
The image ID is obtained by finding the integer between the first two underscores ("_") in the file name.
Therefore an image named XXX_0001_XXX.tif will be matched with images in the other channels' subdirectories
matching image ID 1. No other part of the file name is used.


### Dataset Processing
 The simplest usage for this package is to run _multi_spect_reg.py_.
```bash
python3 multispect_reg.py
```
This script contains 3 lines:
```python
from multi_spect_tools import multi_spect_dataset_handling
dataset_handler = multi_spect_dataset_handling.data_set_handler("cfg/reg_config.ini")
dataset_handler.process_all_images(use_init_transform=True, update_from_previous=True)
```
This script will import the necessary libraries, load the configuration found in the 
_cfg/reg_config.ini_ file, and process all matching sets of images found in the given paths.

## Authors
Multispectral Registration was developed by Kostas Alexis and Frank Mascarich at the Autonomous Robots Lab, University of Nevada, Reno.
* [Kostas Alexis](mailto:kalexis@unr.edu)
* [Frank Mascarich](mailto:fmascarich@nevada.unr.edu)
