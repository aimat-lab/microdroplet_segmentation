# Micro-droplet Segmentation in Microscopy Images

Semi-supervised tool for segmentation of micro-droplet arrays in microscopy images. 
This data analysis tool was developed to automatically detect and evaluate droplet stained by fluorescence markers.

# Requirements 

Required packages for this project can be installed with ``pip`` and are listed below. 
The GUI uses ``matplotlib`` and image processing is handled by `opencv` and `skimage`. 
The code is tested on Windows machines to run on lab computers, where python is available in system PATH environment variable!

```shell
python -m pip install opencv-python
python -m pip install openpyxl
python -m pip install matplotlib
python -m pip install scikit-image
```

# Implementation details

The main program of ``main.py``, executed with `RUN.bat` or `python main.py` runs over all images in "input/" folder 
and creates an equally named output folder in "output/". For each image a series of individual python modules is called,
which can also be used independently but may require input from the previous module. 
Modules are ``ImageSelection.py``, ``FindScaleBar.py``, ``GridSegmentation.py`` and ``DropletSeparation.py``.
General-use classes are given in ``grid.py`` and `image.py`. 
Each module opens a "Qt5Cairo" interactive backend of `matplotlib`. Press "enter" to accept and continue.


# Usage

### FindScaleBar.py

<p align="left">
  <img src="https://github.com/aimat-lab/microdroplet_segmentation/blob/master/_images/scale_bar.png" height="80"/>
</p>



