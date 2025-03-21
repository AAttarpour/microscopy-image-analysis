"""
This code is written by Ahmadreza Attarpour, a.attarpour@mail.utoronto.ca

This code processes a Z-stack of microscopy images stored as a series of .tif files.
It zero-pads the images (if necessary) and creates 3D isotropic patches of a specified size (ZxYxX).
The patches are saved in ZYX order with filenames in the format:
    patch_"Z-index"_"Y-index"_"X-index".tif

For example, if the Z-stack has 2100 slices and the patch size is 512 in depth,
the code will produce patches with names like:
    patch_0_0_0.tif, patch_0_0_512.tif, ..., patch_4_46_30.tif

The first number in the filename represents the Z-index, the second represents the Y-index,
and the third represents the X-index.

In this version it has the following options:
1) recieves the brain mask as the input from registration and include the percentage of tissue in each patch in the metadata.json output.
2) compute the brain mask in the code and include the percentage of tissue in each patch in the metadata.json output.


Input:
    1) A directory containing .tiff or .tif light sheet images corresponding to depth (Z-axis).
    2) A directory containing .tiff or .tif brain mask generated by registration (optional)
    3) A directory for saving the generated image patches.

Output:
    Image patches of size ZxYxX, saved as .tif files.
    A metadata.json file containing:
        - Original image dimensions (height, width, depth).
        - A list of patches with their filenames and coordinates in the original volume.
"""
import os
import numpy as np
import tifffile
from skimage.util import view_as_blocks
import json
from joblib import Parallel, delayed
import multiprocessing
import argparse
from pathlib import Path
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from skimage.transform import resize
from skimage.feature import canny
# -------------------------------------------------------
# Argument parser
# -------------------------------------------------------
my_parser = argparse.ArgumentParser(description='Generate patches from Z-stack .tif files.')
my_parser.add_argument('-i', '--input', help='Input directory containing .tiff or .tif slices', required=True)
my_parser.add_argument('-m', '--brain_mask', help='Input directory containing .tiff or .tif brain mask slices if not passed it will compute mask', required=False, default=False)
my_parser.add_argument('-o', '--out_dir', help='Output directory for patches', required=True)
my_parser.add_argument('-c', '--cpu_load', help='Fraction of CPUs to use for parallelization (0-1)', required=False, default=0.7, type=float)
my_parser.add_argument('-p', '--patch_size', help='Patch size (ZxYxX)', required=False, default=256, type=int)
my_parser.add_argument('-e','--brain_mask_erosion_flag', help='a flag to whether erode the brain mask', required=False, default=False, action='store_true')

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
# brain masker 
# -------------------------------------------------------
def brain_masker(img,
                skip_img_intensity_thr,
                downsample_scale_factor=0.01
                ):
    
    # skip the empty slices
    # check if at least 20% of the image is not background
    if np.sum(img > skip_img_intensity_thr) < 0.20 * img.size:

        img_mask = np.zeros_like(img, dtype = np.bool_)

    else:

        # downsample the image
        img_shape = img.shape
        img = resize_func(img, scale_factor=downsample_scale_factor, mode='downsample', interpolation_order=1).astype(np.uint8)

        edges = canny(img, sigma=1.0)  # Detect edges
        edges = binary_dilation(edges, iterations=3)  # Dilate edges to cover the halo
        img_mask = binary_fill_holes(edges)  # Fill the enclosed region

        # upsample the image
        img_mask = resize_func(img_mask.astype(np.bool_), target_shape=img_shape, mode='upsample', interpolation_order=0)

    return img_mask
# -------------------------------------------------------
# Image saver function
# -------------------------------------------------------
def image_saver(patch, brain_mask_patch, patch_coords, out_dir, patch_size):
    """Save a patch and its metadata."""
    patch_filename = f"patch_{patch_coords[0]}_{patch_coords[1]}_{patch_coords[2]}.tif"
    patch_path = os.path.join(out_dir, patch_filename)
    
    # Save the patch
    tifffile.imwrite(
        patch_path,
        patch.astype("uint16"),
        metadata={
            "DimensionOrder": "ZYX",
            "SizeC": 1,
            "SizeT": 1,
            "SizeX": patch_size,
            "SizeY": patch_size,
            "SizeZ": patch_size,
        },
    )
    
    # compute the amount of tissue in the image
    tissue_percent = 100 * np.sum(brain_mask_patch) / (brain_mask_patch.size)

    # Return metadata for stitching
    return {
        'filename': patch_filename,
        'coordinates': patch_coords,
        'tissue_percentage': tissue_percent
    }

# -------------------------------------------------------
# Load a single slice
# -------------------------------------------------------
def load_slice(file_path):
    """Load a single slice from disk."""
    print(f"Loading slice: {file_path}")
    img = tifffile.imread(file_path)
    if img.ndim > 2:
        return img[-1, -1, :, :]
    else:
        return img 

# -------------------------------------------------------
# Main function
# -------------------------------------------------------
def main(args):
    # Get arguments
    input_path = args['input']
    output_dir = args['out_dir']
    patch_size = args['patch_size']
    cpu_load = args['cpu_load']
    brain_mask_path = args['brain_mask']
    brain_mask_erosion = args['brain_mask_erosion_flag']

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of .tif files in the input directory
    img_list_name = sorted([f for f in os.listdir(input_path) if f.endswith(('.tif', '.tiff'))])
    img_list_name.sort()
    print(f"Found {len(img_list_name)} slices in {input_path}.")


    # Get list of .tif files in the brain mask directory
    if brain_mask_path:
        brain_mask_list_name = sorted([f for f in os.listdir(brain_mask_path) if f.endswith(('.tif', '.tiff'))])
        brain_mask_list_name.sort()
        print(f"Found {len(brain_mask_list_name)} slices in {brain_mask_path}.")
        if len(brain_mask_list_name) != len(img_list_name):
            raise ValueError(f"Mismatch: {len(brain_mask_list_name)} brain masks vs {len(img_list_name)} image slices.")
    else:
        print(f"Brain mask path was not passed ... the script will produce it automatically")

    # Load the first slice to get image dimensions
    first_slice = load_slice(os.path.join(input_path, img_list_name[0]))
    img_height, img_width = first_slice.shape
    num_z_slices = len(img_list_name)

    if not brain_mask_path:
        # compute the skip img intensity thr required for mask generation
        skip_img_intensity_thr = np.mean(first_slice[first_slice > 1])
        skip_img_intensity_thr = 3/2 * skip_img_intensity_thr 

    # Initialize metadata dictionary
    metadata = {
        'original_dimensions': {
            'height': img_height,
            'width': img_width,
            'depth': num_z_slices
        },
        'patches': []
    }

    # Process the Z-stack in chunks of patch_size
    for z_start in range(0, len(img_list_name), patch_size):
        z_end = min(z_start + patch_size, len(img_list_name))
        print("---------------------------")
        print(f"Processing Z-stack chunk: {z_start} to {z_end}")

        # Load the current chunk of slices in parallel
        with Parallel(n_jobs=int(cpu_load * multiprocessing.cpu_count())) as parallel:
            z_stack_chunk = parallel(
                delayed(load_slice)(os.path.join(input_path, img_list_name[i]))
                for i in range(z_start, z_end)
            )
        z_stack_chunk = np.stack(z_stack_chunk, axis=0)  # Stack along Z-axis

        # load or create brain mask
        if brain_mask_path:
            # Load the current chunk of slices in parallel
            with Parallel(n_jobs=int(cpu_load * multiprocessing.cpu_count())) as parallel:
                z_stack_brain_mask_chunk = parallel(
                    delayed(load_slice)(os.path.join(brain_mask_path, brain_mask_list_name[i]))
                    for i in range(z_start, z_end)
                )
            # change the brain mask to bool 
            z_stack_brain_mask_chunk = [mask > 0 for mask in z_stack_brain_mask_chunk]

        else:
            print(f"obtaining brain mask for the currect depth: {z_start} to {z_end}")
            z_stack_brain_mask_chunk = [
                brain_masker(z_stack_chunk[z, ...], skip_img_intensity_thr)
                for z in range(z_stack_chunk.shape[0])
                ]

        # erode the mask
        if brain_mask_erosion:
            z_stack_brain_mask_chunk = [binary_erosion(mask, iterations = 3) for mask in z_stack_brain_mask_chunk] 
        z_stack_brain_mask_chunk = np.stack(z_stack_brain_mask_chunk, axis=0)  # Stack along Z-axis

        # Debug: Print indices and filenames to verify order
        print("Indices:", list(range(z_start, z_end)))
        print("Img Filenames:", img_list_name[z_start:z_end])

        if brain_mask_path:
            print("Brain mask Filenames:", brain_mask_list_name[z_start:z_end])

        # Pad the chunk if necessary
        z_pad = (patch_size - z_stack_chunk.shape[0] % patch_size) % patch_size
        y_pad = (patch_size - z_stack_chunk.shape[1] % patch_size) % patch_size
        x_pad = (patch_size - z_stack_chunk.shape[2] % patch_size) % patch_size
        z_stack_padded = np.pad(z_stack_chunk, ((0, z_pad), (0, y_pad), (0, x_pad)), mode='constant')
        brain_mask_z_stack_padded = np.pad(z_stack_brain_mask_chunk, ((0, z_pad), (0, y_pad), (0, x_pad)), mode='constant')
        print(f"Padded Z-stack chunk with shape: {z_stack_padded.shape}")
        print(f"Padded Z-stack brain mask chunk with shape: {brain_mask_z_stack_padded.shape}")


        # Generate patches using view_as_blocks
        patches = view_as_blocks(z_stack_padded, (patch_size, patch_size, patch_size))
        brain_mask_patches = view_as_blocks(brain_mask_z_stack_padded, (patch_size, patch_size, patch_size))
        print(f"Generated patches with shape: {patches.shape}")
        print(f"Generated brain mask patches with shape: {brain_mask_patches.shape}")

        # Save patches and collect metadata
        with Parallel(n_jobs=int(cpu_load * multiprocessing.cpu_count())) as parallel:
            results = parallel(
                delayed(image_saver)(
                    patches[i, j, k],
                    brain_mask_patches[i, j, k],
                    (z_start + i * patch_size, j * patch_size, k * patch_size),
                    output_dir,
                    patch_size
                )
                for i in range(patches.shape[0])
                for j in range(patches.shape[1])
                for k in range(patches.shape[2])
            )
            metadata['patches'].extend(results)

    # Save metadata for stitching
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {len(metadata['patches'])} patches to {output_dir}.")

if __name__ == '__main__':
    args = vars(my_parser.parse_args())
    main(args)