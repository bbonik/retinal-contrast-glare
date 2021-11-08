# retinal-contrast-glare
A set of (Python &amp; Matlab) functions for estimating the impact of **glare on the formed retinal image**, derived from derived from **calibrated measurements of scene radiances**, or luminances. 

The reason for *not* directly using digital camera outputs, is that, **captured camera images do not represent accurate scene radiances**, because the scene information is transformed by the camera’s optical veiling glare, and nonlinear transformations in camera firmware and image software.

![overview](static/overview.png "overview")

The function requires 2 separate inputs:
1. A **2D input map of the scene** (8-bit input image).
2. A look-up-table of **telephotometer readings** of the scene. 

This is done because telephotometers are the only reliable method of capturing accurate luminances without being affected by the effects of glare. In contrast, camera pixels would have been "contaminated" by glare. Based on these 2 inputs, the function computes the input scene luminances.

Then, a glare spread function is used in order to estimate the retinal image derived by the input scene luminaces. The glare spread function is taken from equation (8) of **Vos&van den Berg (1999) CIE standard** and is used to create a 2D convolution kernel. The kernel size is NxN, where N is the largest dimension of the input image map, in order to ensure that every pixel will "affect" every other one in the image. After that, the kernel is convolved with the input luminance image in order to estimate the cummulated contributions of different points of the scene, on the retinal image. The following is an example of a 600x600 glare spread function kernel.

![kernel](static/kernel_3D.png "kernel")

Finally, different visualizations are generated and saved.

![output](static/Fig-retinal-contrast.png "output")



# Contents:
```tree
├── python                           [Directory: Python source code]
│   ├── retinal_contrast.py          [Script to estimate retinal contrast from an input scene map and a telephotometer LUT file] 
│   └── requirements.txt             [Conda environment file for the required version of libraries]
├── matlab                           [Directory: Matlab source code]
│   ├── computeRetinalContrast.m     [Function to estimate retinal contrast from an input scene map and a telephotometer LUT file]
│   ├── visualizeLogImage.m          [Function to visualize a log-encoded image]
│   └── testRetinalContrast.m        [Script example for testing one input scene]
├── static                           [Directory: example output images]
└── data                             [Directory: sample test scenes]
    ├── scene1                       [Directory: example scene map image 1 and corresponding telephotometer LUT file]
    ├── scene2                       [Directory: example scene map image 2 and corresponding telephotometer LUT file]
    ├── scene3                       [Directory: example scene 3 map image and corresponding telephotometer LUT file]
    └── scene4                       [Directory: example scene 4 map image and corresponding telephotometer LUT file]
```


# Dependences
- skimage
- numpy
- imageio
- matplotlib
- scipy


# Python environment
The code is based on Python 3.8. You can generate a new package environment similar to the one that was used to develop this code, buy running the following command line inside the python directory, where the ```requirements.txt``` is located. 

```conda create --name <your_own_environment_name> --file requirements.txt```


# Citation
If you use this code in your research please cite the following paper:   
1. [McCann, J., Vonikakis, V. (2018). Calculating Retinal Contrast from Scene Content: A Program. Frontiers in Psychology, 8, article 2079](https://www.frontiersin.org/articles/10.3389/fpsyg.2017.02079/full)
