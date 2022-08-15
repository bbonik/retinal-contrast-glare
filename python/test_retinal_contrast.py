#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script shows how to use the retinal contrast function on scene luminances.
Once you import the function compute_retinal_contrast, the only thing you need
to do is pass the required parameters and run this script. The filename of the
input map, as well as the filename of the conversion table, should be strings
of the relative path (Linux style), from the directory where this scrip 
resides, to the directory where the input files reside. 

@author: Vassilios Vonikakis

If you use this code in your research, please cite our papers:

McCann, J., Vonikakis, V. (2018). Calculating Retinal Contrast from Scene 
Content: A Program. Frontiers in Psychology, 8, article 2079
https://www.frontiersin.org/articles/10.3389/fpsyg.2017.02079/full

John J. McCann, Vassilios Vonikakis, Alessandro Rizzi (2022). Edges and 
Gradients in Lightness Illusions: Role of Optical Veiling Glare. 
Frontiers in Psychology

"""

from retinal_contrast import compute_retinal_contrast
import matplotlib.pyplot as plt




if __name__ == "__main__":
    
    plt.close('all')  # close any previous image windows
    
    
    # Change the parameters accordingly. 
    # For more information on the parameters take a look above on the 
    # compute_retinal_contrast() function.
    
    compute_retinal_contrast(
        filename_input_map='../data/scene6/map.tiff',
        filename_conversion_table='../data/scene6/LUT.txt',
        path_output = '../outputs/',  # location where outputs will be saved
        age = 25,
        pigmentation_factor = 0.5,
        pixel_size = 0.05005,
        viewing_distance = 600,
        log_range = 2.302,
        verbose = 2
    )