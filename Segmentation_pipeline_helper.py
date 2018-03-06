#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:33:49 2018

@author: trang.le
"""
import os
import skimage
from scipy import ndimage as ndi
from PIL import Image

# function to find recursively all files with specific prefix and suffix in a directory
def find(dirpath, prefix=None, suffix=None, recursive=True):
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



# Segmentation function
# watershed algorithm to segment the 2d image based on foreground and background seeed plus edges (sobel) as elevation map
# returns labeled segmented image
def watershed_lab(image, marker = None):
    
    # determine markers for watershed if not specified
    if marker is None:
        marker = np.full_like(image,0)
        marker[image ==0] = 1 #background
        marker[image > skimage.filters.threshold_otsu(image)] = 2 #foreground nuclei

    # use sobel to detect edge, then smooth with gaussian filter
    elevation_map = skimage.filters.gaussian(skimage.filters.sobel(image),sigma=2)
    
    #segmentation with watershed algorithms
    segmentation = skimage.morphology.watershed(elevation_map, marker)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_obj, num = ndi.label(segmentation)
    
    # remove bordered objects
    bw = skimage.morphology.closing(labeled_obj > 0, skimage.morphology.square(3))
    cleared = skimage.segmentation.clear_border(bw)
    labeled_obj, num = ndi.label(cleared)
    
    # remove too small or too large object 
    output = np.zeros(labeled_obj.shape)
    for region in regionprops(labeled_obj):
        if region.area >= 20000:# <= thres_high:
            # if the component has a volume within the two thresholds,
            # set the output image to 1 for every pixel of the component
            output[labeled_obj == region.label] = 1
                
    labeled_obj, num = ndi.label(output)
                
    return labeled_obj


# Function to resize and pad segmented image, keeping the aspect ratio 
#(maybe keep the relative ratio of the cells to each other as well?)
def resize_pad(image): #input an Image object (PIL)
    
    desired_size = 256
    
    # current size of the image
    old_size=image.size
    # old_size[0] is in (width, height) format
    ratio = 0.5#float(desired_size)/max(old_size) 
    
    # size of the image after reduced by half
    new_size = tuple([int(x*ratio) for x in old_size])

    # resize image
    im = image.resize(new_size, Image.ANTIALIAS)
    
    # create a new blank image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    
    return new_im

