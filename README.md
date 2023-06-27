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

With ``main.py`` the sequence of the following scripts is executed for each image in `./input` .
Note that the scripts may require output from the previous script. To accept and continue to the next step press 'enter'.
Interactive help and key bindings is usually plotted on the left.

### FindScaleBar.py

First is an optional scale bar information. If the scale bar is detected properly, you can accept its value with 'a'.
The information is stored in 'ScaleBar.yaml'. The detection algorithm is a hard-coded simple search routine.

![plot](./docs/scale_bar.png)

### ImageSelection.py

The image selection allows to remove some boundaries and to pick an image selection and to rotate the image if the image was not aligned.
Increasing brightness and contrast is only for show, but does not alter image information. With 'g' you can activate grid lines to guide the eye.

![plot](./docs/position.png)

### GridSegmentation.py

With this module the grid is adjusted to the data. The auto grid should be already quite good. You can reproduce the auto-grid with 'r'.
A simple uniform standard grid can be generated with 'n' in case the auto-grid fails completely.
You can add/remove grid rows/columns with 'i,j,k,l'. Change between add and remove with 'u'. 
When clicking at the grid lines you can change the origin marked as green. You can lock/unlock this by pressing 'm'.
The grid is stretched with 'w,a,s,d' around the origin. You can move the grid with arrow keys.
Removing rows/columns at the origin should not cause an error, just shift the origin to its proper position.
Note that this is not an infinite grid, so the 0th position for row/column is well-defined.

![plot](./docs/grid.png)

### DropletSeparation.py

Lastly the droplets are 

![plot](./docs/segmentation.png)