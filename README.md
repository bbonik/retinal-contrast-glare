# retinal-contrast-glare
A set of (Python &amp; Matlab) functions for estimating the impact of glare on the retinal image.



# Contents:
```tree
│   
├── python                                      [Directory: Python source code]
│   ├── retinal_contrast.py                   [Script to estimate retinal contrast from an input scene map and a telephotometer LUT file] 
│   └── requirements.txt                [Conda environment file for the required version of libraries]
├── matlab                                      [Directory: Matlab source code]
│   ├── computeRetinalContrast.m                   [Function to estimate retinal contrast from an input scene map and a telephotometer LUT file]
│   ├── visualizeLogImage.m                   [Function to visualize a log-encoded image]
│   └── testRetinalContrast.m                [Script example for testing one input scene]
└── data                                        [Directory: data, models and sample test images]
    ├── scene1                    [Directory: example scene map image 1 and corresponding telephotometer LUT file]
    ├── scene2                    [Directory: example scene map image 2 and corresponding telephotometer LUT file]
    ├── scene3                              [Directory: example scene 3 map image and corresponding telephotometer LUT file]
    └── scene4                               [Directory: example scene 4 map image and corresponding telephotometer LUT file]
```
