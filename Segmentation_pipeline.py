#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 12:52:38 2018

@author: trangle
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
from scipy import ndimage as ndi
from PIL import Image
import gzip
import matplotlib.patches as mpatches
from Segmentation_pipeline_helper import find, watershed_lab, watershed_lab2, resize_pad, find_border, pixel_norm, shift_center_mass

##### EXECUTION PIPELINE FOR CELL SEGMENTATION

# Define path to image input directory
#imageinput = "/afs/pdc.kth.se/projects/cellprofiling/projects/integrated_cell/data/nucleoli_data/nucleoli_images/nucleoli_v18_A-431"
imageinput ='/Users/ngoc.le/Desktop/nucleoli_v18_AF22'

# Define desired path to save the segmented images
imageoutput = "/Users/ngoc.le/Desktop/nucleoli_images/AF22_Nummt"

if not os.path.exists(imageoutput):
    os.makedirs(imageoutput)
    
# Make a list of path to the tif.gz files
nuclei = find(imageinput,prefix=None, suffix="_blue.tif.gz",recursive=False) #blue chanel =nu
nucleoli = []
microtubule = []
for f in nuclei:
    f=f.replace('blue','green')
    nucleoli.append(f)
    f=f.replace('green','red')
    microtubule.append(f)

# For each image, import 3 chanels
# Use nuclei as seed, microtubule as edges to segment the image
# Cut the bounding box of each cell (3channels) in the respective image, slack and save
for index,imgpath in enumerate(nuclei):
    
    print("Segmenting image {0}/{1}".format(index+1,len(nucleoli)))
    name0 = (nuclei[index].split("_AF22/"))[1].split("_blue")[0]
    # Unzip .gz file and read content image to img
    try:
        with gzip.open(imgpath) as f:
            nu = plt.imread(f)
            if len(nu.shape) > 2:
                nu=nu[:,:,2]
    except:
        print("%s does not have valid nucleus channel" % name0)
        continue
    
    try:
        with gzip.open(nucleoli[index]) as f:
            org = plt.imread(f)
            if len(org.shape) > 2:
                org=org[:,:,1]
    except:
        print("%s does not have valid nucleoli channel" % name0)
        continue  
    
    try:
        with gzip.open(microtubule[index]) as f:
            mi = plt.imread(f)
            if len(mi.shape) > 2:
                mi=mi[:,:,0]
    except:
        print("%s does not have valid microtubule channel" % name0)
        continue      

    # obtain nuclei seed for watershed segmentation
    seed, num = watershed_lab(nu, rm_border=False)
    
    # segment microtubule image
    marker = np.full_like(seed, 0)
    marker[mi == 0] = 1 #background
#    marker = skimage.morphology.binary_erosion(marker,skimage.morphology.square(3)).astype(int)
    marker[seed > 0] = seed[seed > 0] + 1 #foreground
    labels = watershed_lab2(mi, marker = marker)
    
    #remove all cells where nucleus is touching the border
    labels = labels - 1
    border_indice = find_border(seed)
    mask = np.in1d(labels,border_indice).reshape(labels.shape)
    labels[mask] = 0
    
    #Plotthing the segmentation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(mi)
    for region in regionprops(labels):
        # take regions with large enough areas
        if region.area >= 20000:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    
    # Cut boundingbox
    i=0
    for region in skimage.measure.regionprops(labels):
        i=i+1
        
        # draw rectangle around segmented cell and
        # apply a binary mask to the selected region, to eliminate signals from surrounding cell
        minr, minc, maxr, maxc = region.bbox
                
        # get mask
        mask = labels[minr:maxr,minc:maxc].astype(np.uint8)
        mask[mask != region.label] = 0
        mask[mask == region.label] = 1

        cell_nuclei = pixel_norm(nu[minr:maxr,minc:maxc]*mask)
        cell_nucleoli = pixel_norm(org[minr:maxr,minc:maxc]*mask)
        cell_microtubule = pixel_norm(mi[minr:maxr,minc:maxc]*mask)

        # stack channels
        cell = np.dstack((cell_microtubule,cell_nucleoli,cell_nuclei))         
        cell = (cell*255).astype(np.uint8) #the input file was uint16         

        # align cell to the 1st major axis  
        theta=region.orientation*180/np.pi #radiant to degree conversion
        cell = ndi.rotate(cell, 90-theta)

        # resize images
        fig = resize_pad(cell) # default size is 256x256
        # center to the center of mass of the nucleus
        fig = shift_center_mass(fig)
        fig = Image.fromarray(fig)
        name = "%s_cell%s.%s" % (name0,str(i), "png")
        name = name.replace("/", "_")
        
        savepath= os.path.join(imageoutput, name)
        #plt.savefig(savepath)
        fig.save(savepath)

