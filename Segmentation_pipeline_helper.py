#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:33:49 2018

@author: trang.le
"""
import os
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.measure import regionprops 
from skimage.feature import peak_local_max
from skimage.morphology import watershed, closing, square
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
from PIL import Image
import numpy as np


def find(dirpath, prefix=None, suffix=None, recursive=True):
    """Function to find recursively all files with specific prefix and suffix in a directory
    Return a list of paths
    """

    l=[]
    if not prefix:
        prefix = ''
    if not suffix:
        suffix = ''
    for (folders, subfolders, files) in os.walk(dirpath):
        for filename in [f for f in files if f.startswith(prefix) and f.endswith(suffix)]:
                l.append(os.path.join(folders, filename))
        if not recursive:
            break
    return l



def watershed_lab(image, marker = None, rm_border = False):
    """Segmentation function
    Watershed algorithm to segment the 2d image based on foreground and background seed 
    and use edges (sobel) as elevation map
    return labeled nuclei
    """
    # determine markers for watershed if not specified
    if marker is None:
        marker = np.full_like(image,0)
        marker[image ==0] = 1 #background
        marker[image > threshold_otsu(image)] = 2 #foreground nuclei

    # use sobel to detect edge, then smooth with gaussian filter
    elevation_map = gaussian(sobel(image),sigma=2)
    
    #segmentation with watershed algorithms
    segmentation = watershed(elevation_map, marker)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_obj, num = ndi.label(segmentation)
    
    if rm_border is True:
        # remove bordered objects, now moved to later steps of segmentation pipeline
        bw = closing(labeled_obj > 0, square(3))
        cleared = clear_border(bw)
        labeled_obj, num = ndi.label(cleared)
    
    # remove too small or too large object 
    output = np.zeros(labeled_obj.shape)
    for region in regionprops(labeled_obj):
        if region.area >= 2000:# <= thres_high:
            # if the component has a volume within the two thresholds,
            # set the output image to 1 for every pixel of the component
            output[labeled_obj == region.label] = 1
                
    labeled_obj, num = ndi.label(output)
                
    return labeled_obj, num

def watershed_lab2(image, marker = None):
    """Watershed segmentation with topological distance 
    and keep the relative ratio of the cells to each other
    return labeled cell body, each has 1 nuclei
    """
    distance = ndi.distance_transform_edt(image)
    distance = clear_border(distance, buffer_size=50)
    # determine markers for watershed if not specified
    if marker is None:
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
        marker = ndi.label(local_maxi)[0]

    #segmentation with watershed algorithms
    segmentation = watershed(-distance, marker, mask = image)
                
    return segmentation

 
def resize_pad(image, size = 256): #input an Image object (PIL)
    """Function to resize and pad segmented image, keeping the aspect ratio 
    and keep the relative ratio of the cells to each other
    """
    image = Image.fromarray(image)
    desired_size = size 
    
    # current size of the image
    old_size=image.size
    # old_size[0] is in (width, height) format
    ratio = 0.3 #float(desired_size)/max(old_size) 
    
    # size of the image after reduced by half
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # resize image
    im = image.resize(new_size, Image.ANTIALIAS)
    
    # create a new blank image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    
    return new_im



def find_border(labels, buffer_size=0, bgval=0, in_place=False):
    """Find indices of objects connected to the label image border.
    Adjusted from skimage.segmentation.clear_border()
    Parameters
    ----------
    labels : (M[, N[, ..., P]]) array of int or bool
        Imaging data labels.
    buffer_size : int, optional
        The width of the border examined.  By default, only objects
        that touch the outside of the image are removed.
    bgval : float or int, optional
        Cleared objects are set to this value.
    in_place : bool, optional
        Whether or not to manipulate the labels array in-place.
    Returns
    -------
    out : (M[, N[, ..., P]]) array
        Imaging data labels with cleared borders
    """
    image = labels

    if any( ( buffer_size >= s for s in image.shape)):
        raise ValueError("buffer size may not be greater than image size")

    # create borders with buffer_size
    borders = np.zeros_like(image, dtype=np.bool_)
    ext = buffer_size + 1
    slstart = slice(ext)
    slend   = slice(-ext, None)
    slices  = [slice(s) for s in image.shape]
    for d in range(image.ndim):
        slicedim = list(slices)
        slicedim[d] = slstart
        borders[slicedim] = True
        slicedim[d] = slend
        borders[slicedim] = True

    # Re-label, in case we are dealing with a binary image
    # and to get consistent labeling
    #labels = skimage.measure.label(image, background=0)

    # determine all objects that are connected to borders
    borders_indices = np.unique(labels[borders])
    
    return borders_indices



def pixel_norm(image):
    """Function to normalize pixel value
    """
    
    # background correction: substracting the most populous pixel value
    image = image - np.median(image)
    image[image < 0] = 0
    
    # rescaling image intensity to a value between 0 and 1
    image = (image - image.min())/(image.max() - image.min())
        
    return image
    
def shift_center_mass(image):
    """Function to center images to the Nuclei center of mass
    assuming channel 2 is the nuclei channel
    """
   
    img = np.asanyarray(image)
    cm_nu = ndi.measurements.center_of_mass(img[:,:,2])
    
    Shift = np.zeros_like(img)
    for channel in (0,1,2): 
        im = img[:,:,channel]
        c = [im.shape[0]/2.,im.shape[1]/2.]
        S = np.roll(im, int(round(c[0]-cm_nu[0])) , axis=0)
        S = np.roll(S, int(round(c[1]-cm_nu[1])), axis=1)        
        Shift[:,:,channel] = S
        
    return Shift
