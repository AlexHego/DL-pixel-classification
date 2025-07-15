# DL-pixel-classification
* [Pixel Classification](#pixel-classification)
* [Goals](#goals)
* [Step-by-step tutorial](#step-by-step-tutorial)
  * [I. installation conda](#i-installation-conda)
  * [II. Create and activate a new environment](#ii-create-and-activate-a-new-environment)
  * [III. Installation dependencies](#iii-installation-dependencies)
  * [IV. Extracting Images from CZI Files](#iv-extracting-images-from-czi-files)
* [Main GUI controls](#main-gui-controls)
* [Citation](#citation)



Pixel Classification
------
Pixel classification in biological images involves assigning a class (e.g. vessel, nucleus, background) to each pixel based on its local appearance. This typically starts with manual annotation, where the user draws or colors selected regions (sparse labeling) to provide examples of each class. Tools like **Ilastik** or **Trainable Weka Segmentation** in Fiji, **QuPath** allow biologists to perform such classification using **Random Forest** classifiers. 

![pixelclassifier](https://github.com/user-attachments/assets/f5daa20b-5f1e-4b6c-ab6a-cf5548d1fd7a)

However, Random Forests have limitations, this motivates the shift toward **deep learning**, like **U-Net**.

Goals
------
This GitHub project demonstrates how to train a **2D U-Net** for semantic segmentation using only a few annotated ROIs across slices from a large 3D light sheet images (Lightsheet Z1 from Zeiss). Labels are sparsely drawn in **Napari**, using the **napari-easy-augment-batch-dl** plugin, and augmented to increase data diversity. The trained model is then applied **slice-by-slice** across the full 3D volume using MONAI’s, and the results are visualized in Napari. While a 3D model might improve performance, this 2D approach balances accuracy and resource efficiency, making it accessible and scalable for large imaging datasets.

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

Next, create a folder on your computer, download the following scripts, and place them inside that folder. <br /> [vessels_semantic_framework.py](https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/imagesc/2025_03_19_vessel_3D_lightsheet/vessels_semantic_framework.py)
<br /> [Extract 2D images script](https://github.com/AlexHego/DL-pixel-classification/blob/main/extract%202D%20images.ipynb)
<br /> [Extract 2D images script](https://github.com/AlexHego/DL-pixel-classification/blob/main/extract%202D%20images.ipynb)


IV. Extracting Images from CZI Files
------
Make sure the conda environment is activated before launching Jupyter Notebook. If not start Miniconda prompt and write <br />`conda activate vessels_lightsheet`<br />
When the environement is activate write `jupyter notebook`


Citation
------

This project builds on the work by Brian Northan [True North Intelligent Algorithms](https://github.com/True-North-Intelligent-Algorithms).

Original example: [3D Vessel Segmentation using 2D U-Net](https://github.com/True-North-Intelligent-Algorithms/tnia-python/tree/main/notebooks/imagesc/2025_03_19_vessel_3D_lightsheet)

I reuse and adapt parts of his code and ideas to demonstrate semantic segmentation workflows for my lightsheet imaging.
All credits for the original method and implementation go to the authors.

