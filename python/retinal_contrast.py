# -*- coding: utf-8 -*-
"""
This is a Python translation (end enhancement) of the original Matlab script 
released for the Frontiers paper, in 2018.

This script defines the retinal contrast function. First the parameters are 
set. Then the function computes input scene luminances by combining a 2D input 
map of the scene and a table of telephotometer readings of the scene. This is 
done because telephotometers are the only reliable method of estimating 
accurate luminances without being affected by the effects of glare.

@author: Vassilios Vonikakis

If you use this code in your research, please cite our papers:

McCann, J., Vonikakis, V. (2018). Calculating Retinal Contrast from Scene 
Content: A Program. Frontiers in Psychology, 8, article 2079
https://www.frontiersin.org/articles/10.3389/fpsyg.2017.02079/full

John J. McCann, Vassilios Vonikakis, Alessandro Rizzi (2022). Edges and 
Gradients in Lightness Illusions: Role of Optical Veiling Glare. 
Frontiers in Psychology

"""

from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import numpy as np
import imageio
import time





def pad_image(image, frame_size):
    # Pads values around the input image, by replicating the pixel values of 
    # the outer border (similar to Matlab's 'replicate' option of imfilter).
    
    # empty image with frame of zeros around it
    image_padded = np.zeros(
        [image.shape[0] + (frame_size * 2), 
         image.shape[1] + (frame_size * 2)],
        dtype=float 
        )
    
    # add the pixel values in the center of the image
    image_padded[
        frame_size:frame_size+image.shape[0],
        frame_size:frame_size+image.shape[1]
        ] = image
    
    # replicating upper left corner
    image_padded[
        0:frame_size,
        0:frame_size
        ] = image[0,0]
    
    # replicating upper mid
    image_padded[
        0:frame_size,
        frame_size:frame_size+image.shape[1]
        ] = np.tile(image[0,:], (frame_size,1))
    
    # replicating upper right corner
    image_padded[
        0:frame_size,
        frame_size+image.shape[1]:
        ] = image[0,-1]

    # replicating right mid
    image_padded[
        frame_size:frame_size+image.shape[0],
        frame_size+image.shape[1]:
        ] = np.tile(image[:,-1].reshape(image.shape[0],1), (1,frame_size))
    
    # replicating bottom right corner
    image_padded[
        frame_size+image.shape[0]:,
        frame_size+image.shape[1]:
        ] = image[-1,-1]
    
    # replicating bottom mid
    image_padded[
        frame_size+image.shape[0]:,
        frame_size:frame_size+image.shape[1]
        ] = np.tile(image[-1,:], (frame_size,1))
    
    # replicating bottom left corner
    image_padded[
        frame_size+image.shape[0]:,
        0:frame_size
        ] = image[-1,0]
    
    # replicating left mid
    image_padded[
        frame_size:frame_size+image.shape[0],
        0:frame_size
        ] = np.tile(image[:,0].reshape(image.shape[0],1), (1,frame_size))
    
    return image_padded





def visualize_log_image(
        image_log, 
        log_range
        ):
    '''
    ----------------------------------------------------------------------
                Prepares a log encoded image for visualization
    ----------------------------------------------------------------------
    Preprocessing for displaying a logarithmic-coded image.
    
    INPUTS
    ------
    image_log: numpy array (float)
        Logarithmic (log10) encoding of a [0,1] image. Since the original 
        image is in the interval [0,1], its logarithmic encoding (imageLog) is 
        in the interval (-inf,0].                  
    log_range: float
        range (in log units) that will be applied on the visualization output.
        
    OUTPUTS
    -------
    image_log_display: numpy array (uint8) 
        Visualization output in the interval [0,255] of the logarithmic input 
        image.
    '''

    image_log_display=image_log.copy()
    
    # truncate anything below the output log range [-log_range,0]
    image_log_display[image_log_display < -log_range] = -log_range
    image_log_display = image_log_display + log_range  # [0,log_range]
    image_log_display = image_log_display / log_range  # [0,1]
    image_log_display = np.uint8(image_log_display * 255)  # [0,255]
    return image_log_display





def get_pseudocolor_map(gradients=64):
    '''
    Defining a custom-made pseudocolor visualization map gradients (int) 
    defines the gradient of colors larger number of gradients results to 
    smoother visualizations. This map was selected to visualize luminance 
    gradients with minimal artefacts in the progression from white to black.
    '''
    
    rgb_list=[
        [0, 0, 0], 
        [0.0825, 0.061875, 0], 
        [0.165, 0.12375, 0], 
        [0.2475, 0.1134375, 0], 
        [0.33, 0.103125, 0], 
        [0.4125, 0.07734375, 0], 
        [0.495, 0.0515625, 0], 
        [0.5775, 0.19078125, 0], 
        [0.66, 0.33, 0], 
        [0.7012, 0.28875, 0], 
        [0.7424, 0.2475, 0], 
        [0.7836, 0.20625, 0], 
        [0.8248, 0.165, 0], 
        [0.866, 0.12375, 0], 
        [0.9072, 0.0825, 0], 
        [0.9484, 0.04125, 0], 
        [1, 0, 0], 
        [1, 0, 0.125], 
        [1, 0, 0.25], 
        [1, 0, 0.375], 
        [1, 0, 0.5], 
        [1, 0, 0.625], 
        [1, 0, 0.75], 
        [1, 0, 0.875], 
        [1, 0, 1], 
        [0.91625, 0.04125, 1], 
        [0.8325, 0.0825, 1], 
        [0.74875, 0.12375, 1], 
        [0.665, 0.165, 1], 
        [0.58125, 0.20625, 1], 
        [0.4975, 0.2475, 1], 
        [0.41375, 0.28875, 1], 
        [0.33, 0.33, 1], 
        [0.35125, 0.41375, 1], 
        [0.3725, 0.4975, 1], 
        [0.39375, 0.58125, 1], 
        [0.415, 0.665, 1], 
        [0.43625, 0.74875, 1], 
        [0.4575, 0.8325, 1], 
        [0.47875, 0.91625, 1], 
        [0.5, 1, 1], 
        [0.5, 1, 0.9375], 
        [0.5, 1, 0.875], 
        [0.5, 1, 0.8125], 
        [0.5, 1, 0.75], 
        [0.5, 1, 0.6875], 
        [0.5, 1, 0.625], 
        [0.5, 1, 0.5625], 
        [0.5, 1, 0.5], 
        [0.5625, 1, 0.4375], 
        [0.625, 1, 0.375], 
        [0.6875, 1, 0.3125], 
        [0.75, 1, 0.25], 
        [0.8125, 1, 0.19], 
        [0.875, 1, 0.13], 
        [0.925, 1, 0.06], 
        [1, 1, 0], 
        [1, 1, 0.125], 
        [1, 1, 0.25], 
        [1, 1, 0.375], 
        [1, 1, 0.5], 
        [1, 1, 0.625], 
        [1, 1, 0.75], 
        [1, 1, 0.88], 
        [1, 1, 1]
        ]
    
    cmap_pseudocolors = LinearSegmentedColormap.from_list(
        'pseudocolors', 
        rgb_list,
        N=gradients,
        gamma=1.0
        )
    
    return cmap_pseudocolors





def compute_retinal_contrast(
        filename_input_map, 
        filename_conversion_table, 
        log_range, 
        path_output=None,
        age = 25, 
        pigmentation_factor = 0.5, 
        pixel_size = 0.05005, 
        viewing_distance = 600, 
        verbose = 1
        ):
    
    '''
    ----------------------------------------------------------------------
              Compute the retinal contrast of an input image map
    ----------------------------------------------------------------------
    The function uses a glare spread function in order to estimate the retinal 
    image derived from the scene’s array of digits (in file <imput_map>) and 
    calibration data. This combination provides calibrated scene luminance 
    input data. The glare spread function is taken from equation (8) of 
    Vos&van den Berg (1999) CIE standard and is used to create a 2D convolution 
    kernel. After that, the kernel is convolved with the input luminance image 
    in order to estimate the cummulated contributions of different points of 
    the scene, on the retinal image.
        
    INPUTS
    ------
    filename_input_map: string
        Path and filename of the scene luminance map (input image). The input
        image map could be any typical image file (e.g. bmp, jpg, png, tif).
    filename_conversion_table: string
        Path and filename of a text file, which will be used as a LUT to 
        translate the pixel values of the input image to spectrometer 
        measurements. Each line of the file corresponds to one pixel value, 
        without any commas. E.g. if the input image map has 256 values, there
        should be 256 different lines (spectrometer measurements) in text file.
    log_range: float
        Output range for the calculated glair, in log units.
    path_output: string
        Path to where the output images will be stored.
    age: float 
        Age of the observer in years.
    pigmentation_factor: float
        0 for very dark eyes, 
        0.5 for brown eyes, 
        1.0 for blue-green caucasians,
        1.2 for blue eyes.
    pixel_size: float 
        Size of pixels in mm.
    viewing_distance: float
        Viewing distance of the observer to the target, in mm.
    verbose: int {0,1,2} 
        Verbosity level. 0: no conslole and no online visualizations (not 
        recommended), 1: only console, 2: console plus online visualizations
    
    OUTPUT
    ------
    retinal_contrast: double precision numpy array HxW
        Numpy array of the linear glair output image.
        
    '''  
       
    
     
    time_start=time.time()
    
    input_map = imageio.imread(filename_input_map)  # load image
    if len(input_map.shape) == 3:
        # make grayscale if not, and color images will loose cd/m2 calibration
        input_map = img_as_ubyte(rgb2gray(input_map))
        
    if verbose > 0:
        print(
            '\nInput map size:', 
            input_map.shape[0], 'H', 
            'x',
            input_map.shape[1], 'W'
            )
        print(
            'Requested log range:', log_range)
    
        
    # load telephotometer LUT txt file
    conversion_table = np.loadtxt(filename_conversion_table)
    
    # computing scene luminances: applying LUT on the scene map
    conversion_table = np.power(10, conversion_table)
    scene_luminance = conversion_table[input_map]
    range_scene_luminance = scene_luminance.max() / scene_luminance.min()
    
    # normalize scene luminance to the maximum (scale by max)
    scene_luminance = scene_luminance / scene_luminance.max()
    
    # compute the log of luminance 
    scene_luminance_log = np.log10(scene_luminance)
    scene_luminance_log_mapped = visualize_log_image(
        image_log=scene_luminance_log, 
        log_range=log_range
        )
    
    if verbose > 0:
        print('Scene luminance statistics:')
        print('    max =', scene_luminance.max())
        print('    min =', scene_luminance.min())
        print('    mean =', scene_luminance.mean())
        print('    lin_range(max/min) =', range_scene_luminance)
        print('    log_range(max-min) =', np.max(scene_luminance_log) - 
                                          np.min(scene_luminance_log))
        
    
    #------------------------------------------------- Calculate filter kernel
    '''
    Calculating the convolution kernel from the glare spread function 
    according to equation (8) of Vos&van den Berg (1999) CIE standard.
    '''
    
    radius = max(scene_luminance.shape)
    filter_kernel = np.zeros(
        [2 * radius + 1, 2 * radius + 1], 
        dtype=float
        )
    
    for i in range(2 * radius + 1):
        
        if verbose > 0:
            progress = int((i*100)/(2*radius))
            print(
                '\r' + 'Calculating filter kernel of ' + 
                str(filter_kernel.shape) + ' ' +
                str(progress) +'%', end=''
                ) 
        
        for j in range(2 * radius + 1):
            
            distance = pixel_size * np.sqrt(
                (i - (radius + 1)) ** 2 + (j - (radius + 1)) ** 2
                )
            # glare angle theta
            th = np.rad2deg(np.arctan(distance / viewing_distance))
            
            filter_kernel[i,j] = (
                (1 - 0.008 * (age/70)**4) *
                (9.2e6 / (1 + (th/0.0046)**2)**1.5 + 
                 1.5e5 / (1 + (th/0.045)**2)**1.5) + 
                (1 + 1.6 * (age/70)**4) * 
                ((400 / (1 + (th/0.1)**2) + 3e-8*th**2) +
                 pigmentation_factor *
                 (1300 / (1 + (th/0.1)**2)**1.5 + 
                  0.8 / (1 + (th/0.1)**2)**0.5)) + 
                2.5e-3 * pigmentation_factor
                )
            
            # correction for flat target instead of sphere
            filter_kernel[i,j] = filter_kernel[i,j] * np.cos(np.deg2rad(th))
            
    
    # normalize filter kernel to avoid adding a DC constant during convolution
    filter_kernel = filter_kernel / np.sum(filter_kernel)


    #-------------------------------------------- Perform the actual filtering
    
    
    # add padding to the image by replicating the values of the outer pixels
    scene_luminance_padded = pad_image(
        image=scene_luminance, 
        frame_size=radius
        )
    
    if verbose > 0:
        print ('\nFiltering padded image', scene_luminance_padded.shape, 
               ' with kernel', filter_kernel.shape, 
               '(this may take time for larger maps)...')
              
    # quick convolution in the frequency domain
    retinal_contrast = fftconvolve(
        in1=scene_luminance_padded, 
        in2=filter_kernel, 
        mode='valid'  # keep only the main values without the surrounding frame
        )
    

    #--------------------------------------------- Estimating retinal contrast
    
    # estimate the log of the retinal contrast image
    retinal_contrast_log = np.log10(retinal_contrast)
    retinal_contrast_log_mapped = visualize_log_image(
        image_log=retinal_contrast_log, 
        log_range=log_range  # range requested by the user (input)
        )
    
    # actual log range of the retinal contrast image
    range_retinal_contrast_log = (np.max(retinal_contrast_log) - 
                                  np.min(retinal_contrast_log))
    
    # statistics of the retinal contrast image
    if verbose > 0:
        print('Retinal contrast statistics:')
        print('    max =', retinal_contrast.max())
        print('    min =', retinal_contrast.min())
        print('    mean =', retinal_contrast.mean())
        print('    lin_range(max/min) =', 
              retinal_contrast.max() / retinal_contrast.min())
        print('    log_range(max-min) =', range_retinal_contrast_log)
    
    
    
    #---------------------------------------------------------- Saving outputs
    
    if verbose > 0:
        print('Saving images')
    
    # preparing the output folder
    if path_output is None: 
        path_output = ''  # if path is not given, use same directory
    elif path_output[-1] != '/':
        path_output += '/'  # if path does not have slash, add one
    
    # if output path does not exist, create it
    if Path(path_output).is_dir() is False:
        Path(path_output).mkdir(parents=True, exist_ok=True)
    
    

    # write output images
    imageio.imwrite(
        uri=f'{path_output}scene_luminance_log_mapped[range={log_range}].tiff', 
        im=scene_luminance_log_mapped
        )
    imageio.imwrite(
        uri=f'{path_output}retinal_contrast_log_mapped[range={log_range}].tiff', 
        im=retinal_contrast_log_mapped
        )
    
    # create pseudocolor map
    cmap_pseudocolors = get_pseudocolor_map(  
        gradients=64
        )  
    
    # saving retinal contrast with pseudocolors
    plt.ioff()  # in order not to display the image
    fig = plt.figure()
    plt.imshow(
        retinal_contrast_log_mapped, 
        cmap=cmap_pseudocolors, 
        vmin=0, 
        vmax=255
        )
    plt.title('Retinal Contrast Log Pseudocolors (range = ' + str(log_range) + ')')
    plt.colorbar()
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    fig.savefig(
        f'{path_output}retinal_contrast_log_pseudocolors_range={log_range}.png', 
        bbox_inches='tight'
        )
    plt.close(fig)
    
    
    
    # saving scene luminance with pseudocolors
    plt.ioff()  # in order not to display the image
    fig = plt.figure()
    plt.imshow(
        scene_luminance_log_mapped, 
        cmap=cmap_pseudocolors, 
        vmin=0, 
        vmax=255
        )
    plt.title('Scene Luminance Log Pseudocolors (range = ' + str(log_range) + ')')
    plt.colorbar()
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    fig.savefig(
        f'{path_output}scene_luminance_log_pseudocolors_range={log_range}.png', 
        bbox_inches='tight'
        )
    plt.close(fig)
     
        
    
    #---------------------------------------------------------- Visualizations
    

    if verbose > 1:
           
        # 3D filter kernel
        x = range(filter_kernel.shape[1])
        y = range(filter_kernel.shape[0])
        [X,Y] = np.meshgrid(x,y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            X,
            Y,
            np.log10(filter_kernel), 
            cmap='jet', 
            linewidth=0, 
            rstride=10,  # subsampling ratio
            cstride=10  # subsampling ratio
        )
        plt.xlabel('X dimension')
        plt.ylabel('Y dimension')
        ax.set_zlabel('Log units')
        plt.title('Glair filter kernel logarithmic \n' + 
                  str(filter_kernel.shape[0]) + 
                  ' x ' + 
                  str(filter_kernel.shape[1])
        )
        plt.tight_layout()
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        
    
        # for scene luminance 
        fig1, ax = plt.subplots(nrows=2, ncols=2)
        
        plt.subplot(2, 2, 1)
        plt.imshow(input_map, cmap='gray')
        plt.title('Input Map \n(pixels values)')
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.imshow(scene_luminance_log, cmap='gray')
        plt.title('Input Scene Luminance Log')
        plt.colorbar()
        
        plt.subplot(2, 2, 3)
        plt.imshow(scene_luminance, cmap='gray')
        plt.title('Input Scene Luminance \n(normalized telephotometer values)')
        plt.colorbar()
        
        plt.subplot(2, 2, 4)
        plt.imshow(scene_luminance_padded, cmap='gray')
        plt.title('Input Scene Luminance Padded \n(normalized telephotometer values)')
        plt.colorbar()
        
        fig1.set_size_inches(16, 9, forward=True)
        plt.tight_layout()
        plt.show()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        # uncomment to save the figure
        # fig1.savefig(
        #     f'{path_output}Fig-scene-luminance.png', 
        #     bbox_inches='tight'
        #     )
    
    
        # for retinal contrast  
        fig2, ax = plt.subplots(nrows=1, ncols=3)
        
        plt.subplot(1, 3, 1)
        plt.imshow(
            retinal_contrast, 
            cmap='gray', 
            vmin=0, 
            vmax=1
            )
        plt.title('Retinal Contrast Linear Output')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(
            retinal_contrast_log_mapped, 
            cmap=cmap_pseudocolors, 
            vmin=0, 
            vmax=255
            )
        plt.title(
            'Retinal Contrast Pseudocolors \n(range = ' + str(log_range) + ')'
            )
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(
            retinal_contrast_log, 
            cmap='gray', 
            vmin=retinal_contrast_log.min(), 
            vmax=0
            )
        plt.title(
            'Retinal Contrast Log Output \n(range = ' + str(log_range) + ')'
            )
        plt.colorbar()
        
        fig2.set_size_inches(16, 9, forward=True)
        plt.tight_layout()
        plt.show()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        # uncomment to save the figure
        # fig2.savefig(
        #     f'{path_output}Fig-retinal-contrast.png', 
        #     bbox_inches='tight'
        #     )
    

        print('Finished in:', time.time()-time_start, 'sec')
    
    return retinal_contrast
        




if __name__ == "__main__":
    
    plt.close('all')  # close any previous image windows
    
    
    # Change parameters accordingly. 
    # For more information on the parameters take a look above on the 
    # compute_retinal_contrast() function.
    
    compute_retinal_contrast(
        filename_input_map='../data/scene1/map.tiff',
        filename_conversion_table='../data/scene1/LUT.txt',
        path_output = '../outputs/',  # location where outputs will be saved
        age = 25,
        pigmentation_factor = 0.5,
        pixel_size = 0.1664,
        viewing_distance = 360,
        log_range = 5.4,
        verbose = 2
    )

