"""
This code is written by Ahmadreza Attarpour (a.attarpour@mail.utoronto.ca) 
This code reads the raw signal lsm data and create a brain template mask for the data

- main input is the directory containing the raw 2D slices of the lsm data
- output is the directory containing the mask of the brain

it uses cannny edge detection to detect the edges of the brain and then fill the enclosed region to create the mask

"""


import tifffile 
import numpy as np
from joblib import Parallel, delayed, parallel_config
import multiprocessing
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
import os
import argparse
from skimage.filters import threshold_otsu
from skimage.exposure import exposure, adjust_gamma
import scipy.ndimage as ndimage
from skimage.transform import resize
from skimage.morphology import remove_small_objects
from skimage.feature import canny


# -------------------------------------------------------
# create parser
# -------------------------------------------------------
my_parser = argparse.ArgumentParser(description='Working directory')

# Add the arguments
my_parser.add_argument('-i','--input', help='input directory containing tif/tiff raw 2D slices', required=True)
my_parser.add_argument('-o','--out_dir', help='path of output directory', required=True)
my_parser.add_argument('-c','--cpu_load', help='fraction of cpus to be used for parallelization between 0-1', required=False, default=0.7, type=float)
my_parser.add_argument('-g','--gamma', help='gamma for gamma correction algorithm', required=False, default=0.3, type=float)
my_parser.add_argument('-s','--down_scale', help='downsample scale factor', required=False, default=0.05, type=float)


# -------------------------------------------------------
# save function
# -------------------------------------------------------
def save_tiff(img, name, dir, type):

    tifffile.imwrite(os.path.join(dir, name), img.astype(type), metadata={'DimensionOrder': 'ZYX',
                                                                            'SizeC': 1,
                                                                            'SizeT': 1,
                                                                            'SizeX': img.shape[0],
                                                                            'SizeY': img.shape[1],
                                                                            'SizeZ': 1})
# -------------------------------------------------------
# computes the threshold for empty slices
# -------------------------------------------------------
def empty_slice_thr_calculator(
        file_name_path
        ):
    # file_name_path is the first slice of the 3D image (should be all background)
    img = tifffile.imread(file_name_path).astype(np.uint8)

    return np.mean(img[img > 1])

# -------------------------------------------------------
# function to upsample or downsample the image
# -------------------------------------------------------
def resize_func(img, scale_factor=None, target_shape=None, mode='downsample', interpolation_order=1):
    """
    Resize an image by downsampling or upsampling.

    Parameters:
        img (numpy.ndarray): Input image.
        scale_factor (float): Scaling factor for downsampling. Ignored if `target_shape` is provided.
        target_shape (tuple): Target shape (height, width) for upsampling. Required if mode is 'upsample'.
        mode (str): Resizing mode. Options: 'downsample', 'upsample'.
        interpolation_order (int): Interpolation order. 
                                   - 0: nearest-neighbor
                                   - 1: bilinear
                                   - 3: cubic

    Returns:
        numpy.ndarray: Resized image.
    """
    if mode not in ['downsample', 'upsample']:
        raise ValueError("Invalid mode. Choose 'downsample' or 'upsample'.")

    if mode == 'upsample' and target_shape is None:
        raise ValueError("For upsampling, `target_shape` must be provided.")

    if mode == 'downsample' and scale_factor is None:
        raise ValueError("For downsampling, `scale_factor` must be provided.")

    if mode == 'downsample':
        # Downsample the image
        new_shape = (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor))
        resized_img = resize(img, new_shape, order=interpolation_order, anti_aliasing=True, preserve_range=True)
    else:
        # Upsample the image
        resized_img = resize(img, target_shape, order=interpolation_order, anti_aliasing=False, preserve_range=True)

    # Ensure the output has the same data type as the input
    resized_img = resized_img.astype(img.dtype)

    return resized_img

# -------------------------------------------------------
# function to enhance contrast
# -------------------------------------------------------
def enhance_contrast_stretching(img, saturated_pixels=0.35):
    """
    Enhance contrast using histogram stretching.

    Parameters:
        img (numpy.ndarray): Input image.
        saturated_pixels (float): Percentage of pixels to saturate (clip) at the low and high ends.

    Returns:
        numpy.ndarray: Contrast-enhanced image.
    """
    # Calculate the intensity values corresponding to the saturated pixels
    low, high = np.percentile(img, [saturated_pixels, 100 - saturated_pixels])

    # Stretch the intensity range
    img_stretched = exposure.rescale_intensity(img, in_range=(low, high))

    return img_stretched
# -------------------------------------------------------
# overall function
# -------------------------------------------------------
def brain_masker(idx,
                file_names_paths,
                out_dir,
                skip_img_intensity_thr,
                downsample_scale_factor=0.05,
                gamma=0.3):

    img = tifffile.imread(file_names_paths[idx]).astype(np.uint8)

    # skip the empty slices
    # check if at least 10% of the image is not background
    if np.sum(img > skip_img_intensity_thr) < 0.25 * img.shape[0] * img.shape[1]:

        print(f"skipping {os.path.basename(file_names_paths[idx])} due to containing only background")

        img_mask = np.zeros_like(img, dtype = np.bool_)
        save_tiff(img_mask, 'mask_' + os.path.basename(file_names_paths[idx]), out_dir, np.bool_)

    else:

        # downsample the image
        img_shape = img.shape
        img = resize_func(img, scale_factor=downsample_scale_factor, mode='downsample', interpolation_order=1).astype(np.uint8)

        # gamma correction
        # img = adjust_gamma(img, gamma=gamma, gain=1)
        # img = enhance_contrast_stretching(img, saturated_pixels=0.35)

        # Gaussian smoothing
        # img = ndimage.gaussian_filter(img, sigma=5)
        # img = ndimage.median_filter(img, size=3)
        # img = ndimage.gaussian_filter(img, sigma=2)

        # # Otsu thresholding
        # thr = threshold_otsu(img)
        # img_mask = img > thr

        edges = canny(img, sigma=1.0)  # Detect edges
        edges = binary_dilation(edges, iterations=3)  # Dilate edges to cover the halo
        img_mask = binary_fill_holes(edges)  # Fill the enclosed region

        # filling the holes in the foreground
        # img_mask = binary_fill_holes(img_mask)

        # erosion and dilation to remove the small detected foreground in the backgroun
        # img_mask = binary_erosion(img_mask, iterations = 3)
        # img_mask = binary_dilation(img_mask, iterations = 3)
        # img_mask = remove_small_objects(img_mask, min_size=500)

        # upsample the image
        full_mask = resize_func(img_mask.astype(np.bool_), target_shape=img_shape, mode='upsample', interpolation_order=0)

        save_tiff(full_mask, 'mask_' + os.path.basename(file_names_paths[idx]), out_dir, np.bool_)

        print(f"mask for {os.path.basename(file_names_paths[idx])} is saved!")

def main(args):

    # get the arguments
    cpu_load = args['cpu_load']
    input_names_paths = args['input']
    out_dir = args['out_dir']
    gamma = args['gamma']
    down_scale = args['down_scale']

    print(f"the following params will be used: \n",
          f"input dir: {input_names_paths} \n",
          f"output dir: {out_dir} \n",
          f"gamma: {gamma}")

    # create out dir
    isExist = os.path.exists(out_dir)
    if not isExist: os.mkdir(out_dir)

    # List all files in the input path
    file_names = os.listdir(input_names_paths)
    input_names_paths = [os.path.join(input_names_paths, file) for file in file_names if file.endswith('.tiff') or file.endswith('.tif')]
    input_names_paths.sort()

    # get the number of cpus
    cpus = multiprocessing.cpu_count()
    ncpus = int(cpu_load * cpus)


    skip_img_intensity_thr = empty_slice_thr_calculator(input_names_paths[0])
    skip_img_intensity_thr = 3/2 * skip_img_intensity_thr 
    print(f"skip_img_intensity_thr is {skip_img_intensity_thr}")

    # parallel computng
    with parallel_config(backend = "threading", n_jobs = ncpus):

        Parallel()(delayed(brain_masker)(

                idx,
                input_names_paths,
                out_dir,
                skip_img_intensity_thr,
                down_scale,
                gamma,
            
            )
            
            for idx in range(len(input_names_paths))

            )
        
if __name__ == '__main__':

    # Execute the parse_args() method
    args = vars(my_parser.parse_args())
    main(args)

