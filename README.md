# DL-pixel-classification
* [Pixel Classification](#pixel-classification)
* [Goals](#goals)
* [Step-by-step tutorial](#step-by-step-tutorial)
  * [I. Installation Fiji and Cellpose](#i-installation-fiji-and-cellpose)
  * [II. Starting Cellpose GUI on PC](#ii-starting-cellpose-gui-on-pc)
  * [III. Using the Cellpose GUI](#iii-using-the-cellpose-gui)
* [Main GUI controls](#main-gui-controls)
* [Citation](#citation)



Pixel Classification
------
Pixel classification in biological images involves assigning a class (e.g. vessel, nucleus, background) to each pixel based on its local appearance. This typically starts with manual annotation, where the user draws or colors selected regions (sparse labeling) to provide examples of each class. Tools like **Ilastik** or **Trainable Weka Segmentation** in Fiji, **QuPath** allow biologists to perform such classification using **Random Forest** classifiers. 

![pixelclassifier](https://github.com/user-attachments/assets/f5daa20b-5f1e-4b6c-ab6a-cf5548d1fd7a)

However, Random Forests have limitations, this motivates the shift toward **deep learning**, like **U-Net**.

Goals
------
This GitHub project demonstrates how to train a **2D U-Net** for semantic segmentation using only a few annotated ROIs across slices from a large 3D light sheet image. Labels are sparsely drawn in **Napari**, using the **napari-easy-augment-batch-dl** plugin, and augmented to increase data diversity. The trained model is then applied **slice-by-slice** across the full 3D volume using MONAI’s, and the results are visualized in Napari. While a 3D model might improve performance, this 2D approach balances accuracy and resource efficiency, making it accessible and scalable for large imaging datasets.

Step-by-step tutorial
------

I. installation Conda
------
Install Miniconda [link](https://www.anaconda.com/download/success) and check the [documentation](https://www.anaconda.com/docs/main) for more informations.


Citation
------

This project builds on the work by Brian Northan [True North Intelligent Algorithms](https://github.com/True-North-Intelligent-Algorithms).

Original example: [3D Vessel Segmentation using 2D U-Net](https://github.com/True-North-Intelligent-Algorithms/tnia-python/tree/main/notebooks/imagesc/2025_03_19_vessel_3D_lightsheet)

I reuse and adapt parts of his code and ideas to demonstrate semantic segmentation workflows for my lightsheet imaging.
All credits for the original method and implementation go to the authors.

