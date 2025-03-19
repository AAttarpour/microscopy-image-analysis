# Microscopy Image Analysis

This repository contains a collection of scripts for processing and analyzing microscopy images especially large tera-voxel size (trillions of voxels) light sheet fluorescence microscopy (LSFM) images. The scripts handle various stages of the image processing pipeline, including data handling, preprocessing, and postprocessing.  

Author: **Ahmadreza Attarpour**  
Contact: [a.attarpour@mail.utoronto.ca](mailto:a.attarpour@mail.utoronto.ca)  

## 🚀 Usage
These scripts have been integrated into larger end-to-end pipelines such as [ACE](https://www.nature.com/articles/s41592-024-02583-1), [MIRACL](https://miracl.readthedocs.io/en/latest/index.html), and MAPL3 (currently in progress). Stay tuned for more scripts.

## 📁 Directory Structure

- **data-handling/**: Scripts for generating image patches and creating brain masks.
- **preprocessing/**: Scripts for correcting light sheet artifacts and enhancing images.
- **postprocessing/**: Scripts for reconstructing whole-brain images and performing morphological filtering.

---

## 🛠 Data Handling

### **1️⃣ Generate Patch Script**
This script processes a Z-stack of microscopy images stored as **.tif** files. It creates 3D isotropic patches of a specified size (**Z×Y×X**) and saves them with filenames in the format:

```python
patch_"Z-index""Y-index""X-index".tif
```
📌 **Features**:
- Accepts an optional **brain mask** from registration step and includes the **tissue percentage** in `metadata.json`.
- Can compute the **brain mask** internally if not provided using Canny edge detection algorithm.

🔹 **Inputs**:
1. Directory containing `.tif` light sheet images (Z-stack).  
2. Directory containing `.tif` brain masks (optional).  
3. Output directory for storing generated patches.  

🔹 **Outputs**:
- **Image patches** of size `Z×Y×X`, saved as `.tif` files. The saving process is **parallelized**. 
- **Metadata.json** containing:
  - Original image dimensions.
  - List of patches with their filenames and coordinates.

---

### **2️⃣ Brain Masker**
This script generates a **brain template mask** from raw LSFM data. It detects the brain boundary using **Canny edge detection** and fills the enclosed region to create the mask.

🔹 **Inputs**:
- Directory containing raw 2D `.tif` slices of LSFM data.  

🔹 **Outputs**:
- Directory containing the generated **brain mask**.

---

## 🔧 Preprocessing

### **1️⃣ Preprocessing Script (Parallelized)**
This script preprocesses **raw LSFM images** for downstream analysis, including **light sheet correction** and **pseudo-deconvolution**.

📌 **Processing Steps**:
1. **Light Sheet Correction**:
   - Inspired by [TubeMap](https://christophkirst.github.io/ClearMap2Documentation/html/tubemap.html).
   - Removes light sheet artifacts using percentile-based background estimation.
   - Downsampling and a **3D kernel** are used to accelerate computation.

2. **Pseudo-Deconvolution**:
   - Identifies **high-intensity** voxels (95th percentile) and blurs them using a **3D Gaussian filter**.
   - Subtracts the blurred result while preserving high-intensity voxel values.

🔹 **Inputs**:
1. Directory containing **raw** `.tif` 3D image patches.  
2. Output directory for **preprocessed** patches.  

🔹 **Outputs**:
- Directory containing **processed** image patches.

---

## 📤 Postprocessing

### **1️⃣ Patch Stacking**
This script reconstructs a **whole-brain 3D dataset** from processed image patches.

📌 **Features**:
- Reads **model outputs** and reconstructs the original volume.
- Ensures missing patches are filled with **empty patches**.
- Supports **parallelization** for faster processing.

🔹 **Inputs**:
- **input**: Directory containing processed patches.  
- **out_dir**: Output directory for the final stacked **Z-stack**.  
- **raw_dir**: Directory with original `.tif` slices for reference dimensions.  
- **cpu_load**: Fraction of CPUs to use for parallelization (0-1).  
- **metadata_path**: Path to `metadata.json` (original patch metadata).  
- **dtype**: Output data type (e.g., `uint16`, `bool`).  

🔹 **Outputs**:
- **Final stitched Z-stack**, saved as `.tif` slices.

---

### **2️⃣ Skeletonization & Morphological Filtering (Parallelized)**
This script **skeletonizes** a probability map and removes artifacts based on shape and volume.

📌 **Processing Steps**:
1. **Load** the probability map (deep learning output).
2. **Binarize** at multiple thresholds (0.1 - 0.9).
3. Apply **Medial Axis Transform** to compute the **distance transform**.
4. Sum up all distance transform maps.
5. Detect **ridge structures** using `peak_local_max()`.
6. Filter objects using:
   - **Volume threshold**.
   - **Orientation threshold**.
   - **Eccentricity threshold** (roundness of objects).

🔹 **Inputs**:
- **input**: Directory containing `.tif` probability maps.  
- **out_dir**: Output directory for skeletonized images.  
- **remove_small_obj_thr**: Threshold for removing small objects (voxel count).  
- **cpu_load**: Fraction of CPUs for parallel processing (0-1).  
- **dilate_distance_transform_flag**: Option to dilate distance transform.  
- **eccentricity_thr**: Eccentricity filter (0-1; circles = 0).  
- **orientation_thr**: Orientation filter (degrees).  

🔹 **Outputs**:
- Directory containing **skeletonized binary maps**.

---



## 📜 License
This repository is intended for **research and educational purposes**. Contact the author for inquiries about usage.

---








