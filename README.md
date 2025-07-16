# DL-pixel-classification
* [Pixel Classification](#pixel-classification)
* [Goals](#goals)
* [Step-by-step tutorial](#step-by-step-tutorial)
  * [I. installation conda](#i-installation-conda)
  * [II. Create and activate a new environment](#ii-create-and-activate-a-new-environment)
  * [III. Installation dependencies](#iii-installation-dependencies)
  * [IV. Organizing and preparing data](#iv-organizing-and-preparing-data)
  * [V. Extracting Images from CZI Files](#v-extracting-images-from-czi-files)
  * [VI. Napari Viewer and plugin interface](#vi-napari-viewer-and-plugin-interface)
  * [VII. Sparse annotations and training](#vii-sparse-annotations-and-training)
* [Results](#results)
* [Citation](#citation)



Pixel Classification
------
Pixel classification in biological images involves assigning a class (e.g. vessel, nucleus, background) to each pixel based on its local appearance. This typically starts with manual annotation, where the user draws or colors selected regions (sparse labeling) to provide examples of each class. Tools like **Ilastik** or **Trainable Weka Segmentation** in Fiji, **QuPath** allow biologists to perform such classification using **Random Forest** classifiers. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/f5daa20b-5f1e-4b6c-ab6a-cf5548d1fd7a">
</p>


However, Random Forests have limitations, this motivates the shift toward **deep learning**, like **U-Net**.

Goals
------
This GitHub project demonstrates how to train a **2D U-Net** for semantic segmentation using only a few annotated ROIs across slices from a large 3D light sheet images (Lightsheet Z1 from Zeiss). Labels are sparsely drawn in **Napari**, using the **napari-easy-augment-batch-dl** plugin, and augmented to increase data diversity. The trained model is then applied **slice-by-slice** across the full 3D volume using MONAI’s, and the results are visualized in Napari. While a 3D model might improve performance, this 2D approach balances accuracy and resource efficiency, making it accessible and scalable for large imaging datasets.

**Note:**
This GitHub project only works with 3D `.czi` images that contain a single channel.

Step-by-step tutorial
------
For windows

I. installation conda
------
Install Miniconda [link](https://www.anaconda.com/download/success) and check the [documentation](https://www.anaconda.com/docs/main) for more informations.
During the installation, check the box "Add Anaconda/Miniconda to my PATH environment variable".

II. Create and activate a new environment
------
Start Miniconda prompt and write <br />   `conda create -n vessels_lightsheet -c conda-forge python=3.11` <br /> then <br />    `conda activate vessels_lightsheet`<br />

III. Installation dependencies
------
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`<br />
`pip install "napari[all]"`<br />
`pip install albumentations matplotlib scipy tifffile czifile `<br />
`pip install --upgrade git+https://github.com/Project-MONAI/MONAI `<br />
`pip install --upgrade git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git`<br />
`pip install --upgrade git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git`<br />
`pip install notebook`<br />

When launching Jupyter Notebook, it often opens in a default directory like `C:\Users\YourUsername`. To avoid navigation issues, it's recommended to create your project folder directly in this default location.Ten, download the following scripts, and place them inside that folder. <br /> [vessels_semantic_framework.py](https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/imagesc/2025_03_19_vessel_3D_lightsheet/vessels_semantic_framework.py)
<br /> [Extract 2D images and crop script](https://github.com/AlexHego/DL-pixel-classification/blob/main/01_extract%202D%20images%20and%20crop.ipynb)
<br /> [Training script](https://github.com/AlexHego/DL-pixel-classification/blob/main/02_Training_sparse_label.ipynb)

IV. Organizing and preparing data
------
Place your `.czi` image files (with only one channel) in a folder named `data`. Please follow this structure.

```plaintext
parent_path/
├── data/
│   ├── image1.czi
│   ├── image2.czi
│   └── ...
```
Then the script will automatically create two new folders:

- **extracted_tiff**: contains the extracted 2D slices (every 25 planes) from the `.czi` files.
- **cropped_tiff**: contains the final cropped 2D images, resized to ensure the same dimensions in both **x** and **y**.

```plaintext
parent_path/
├── data/
│   ├── image1.czi
│   ├── image2.czi
│   └── ...
├── extracted_tiff/
│   ├── image1_z000.tif
│   ├── image1_z025.tif
│   └── ...
├── cropped_tiff/
│   ├── image1_z000.tif
│   ├── image1_z025.tif
│   └── ...
```
V. Extracting Images from CZI Files
------

Before launching Jupyter Notebook, ensure that the conda environment is activated. If it isn’t, open the Miniconda Prompt and run: <br />  `conda activate vessels_lightsheet`. <br />  Once the environment is activated, start Jupyter Notebook by typing: `jupyter notebook`.
<br /> Then, navigate to the folder where your scripts are saved and open the notebook titled "Extract 2D images and crop". Launch the script from there. (**Note** : You will have to change the parent_path) 

VI. Napari Viewer and plugin interface 
------
Open the notebook **`02_Training_sparse_label.ipynb`** and run the script. This will launch **Napari**, where you will draw sparse annotations and begin the training process.
## Napari Viewer tutorial:
<img width="250" alt="image" src="https://github.com/user-attachments/assets/52abe2f8-8800-4d12-b399-0c780544ef0e" align="right">

link to the [documentation](https://napari.org/dev/tutorials/fundamentals/viewer.html)
<br />
**Left Panel – Image Viewer & Layer Controls**

Used to adjust display settings of the selected layer:

- **Opacity**: Controls layer transparency.
- **Blending**: Current mode: `translucent_no_depth`.
- **Contrast Limits**: Adjusts brightness range.
- **Auto-contrast**:  
  - `once`: Set once when image is loaded.  
  - `continuous`: Updates automatically.
- **Gamma**: Controls image intensity midtones.
- **Colormap**: Current colormap is `gray`.
- **Interpolation**: Rendering mode (`nearest` selected).

**Layer List**
This section lists all visible layers in the napari viewer:

- `Label box`: Manual box. Where IA check label               
- `predictions_0`: Model-predicted semantic segmentation.    
- `labels_0`: Manual ground truth labels.                  
- `images`: Raw input image currently viewed.              

Note : Use the eye icon to toggle visibility for each layer.

---
<img width="300" alt="image" src="https://github.com/user-attachments/assets/62b3298d-7dd0-4372-9d00-bb1b8d0a58b8" align="right">

##  Right Panel – Plugin Functionalities

**1. Draw Labels**
- `Open image directory…`: Load raw image dataset.
- `Save results…`: Save drawing and prediction

**2. Augment Images**
all the augmentation can be use and check
- `Augment current image`: Apply only to the open image.
- `Augment all images`: Batch augment all loaded images.
- `Delete augmentations`: Remove generated augmented patches.
- `Settings…`: Open advanced augmentation options.

**3. Train / Predict**
Interface to train and use a deep learning segmentation model.

- **Model type**: `Vessels Semantic Model`
- `Load`: Load a pre-trained model.
- `Train network`: Start model training on labeled data.
- `Predict current image`: Run inference on the visible image.
- `Predict all images`: Apply prediction to all images in batch.


VII. Sparse annotations and training
------

### Labeling Guidelines:
This method uses **sparse labeling**, meaning that not every pixel needs to be labeled. However, it is important to label **some background pixels** to differentiate between actual background and unlabeled regions.
If you have two foreground classes, use the following labels:
- `1` → Background
- `2` → Class 1
- `3` → Class 2

Pixels labeled as `0` are considered **unlabeled** and will be **ignored** during training.



Results
------

Below are some example of results. 

**Color legend:**
- 🟢 **Green** – Raw data  
- 🔴 **Red** – Model prediction  
- 🟡 **Yellow** – Overlay (Raw + Prediction)

<p align="center">
  <img src="https://github.com/user-attachments/assets/fc3f62c7-0fcc-4c1b-bed9-8699b274eb02" height="350" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/d6db9482-c442-49fd-88e5-60418ad2de43" height="350" />
</p>




Citation
------
This project builds on the work by Brian Northan [True North Intelligent Algorithms](https://github.com/True-North-Intelligent-Algorithms). I reuse and adapt parts of his code and ideas to demonstrate semantic segmentation workflows for my lightsheet imaging.

If you use it successfully for your research please be so kind to cite this Github: [3D Vessel Segmentation using 2D U-Net](https://github.com/True-North-Intelligent-Algorithms/tnia-python/tree/main/notebooks/imagesc/2025_03_19_vessel_3D_lightsheet)


