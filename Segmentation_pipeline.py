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

##### EXECUTION PIPELINE FOR CELL SEGMENTATION

# Define path to image input directory
imageinput = "nucleoli"#"zprojs_organelle_markerplate4_subset/"

# Define desired path to save the segmented images
imageoutput = "Nuclei_nucleoli/PNG/"


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
    
    # Unzip .gz file and read content image to img
    with gzip.open(imgpath) as f:
        nu = plt.imread(f)
        if len(img.shape) == 3:
            img=img[:,:,2]
            
    with gzip.open(nucleoli[index]) as f:
        org = plt.imread(f)
        if len(org.shape) == 3:
            org=img2[:,:,1]
            
    with gzip.open(microtubule[index]) as f:
        mi = plt.imread(f)
        if len(mi.shape) == 3:
            mi=mi[:,:,0]
    
    # obtain nuclei seed for watershed segmentation
    seed, num = watershed_lab(nu)
    
    # segment microtubule image
    #marker = np.full_like(seed, 0)
    #marker[seed > 0] = seed[seed > 0] + 1 #foreground
    #marker[mi == 0] = 1 #background
    #labels = watershed_lab(mi, marker = marker)
    labels = watershed_lab2(mi, marker = seed)
    
    # Cut boundingbox
    i=0
    for region in skimage.measure.regionprops(labels):
        i=i+1
        
        # draw rectangle around segmented cell and
        # apply a binary mask to the selected region, to eliminate signals from surrounding cell
        minr, minc, maxr, maxc = region.bbox

        mask = labels[minr:maxr,minc:maxc].astype(np.uint8)
        mask[mask != region.label] = 0
        mask[mask == region.label] = 1

        cell_nuclei = nu[minr:maxr,minc:maxc]*mask
        cell_nucleoli = np.full_like(cell_nuclei,0)*mask
        cell_microtubule = mi[minr:maxr,minc:maxc]*mask
        
        # stack channels
        cell = np.dstack((cell_microtubule,cell_nucleoli,cell_nuclei))
        if cell.dtype == 'uint16' :
            cell = (cell/255).astype(np.uint8) #the input file was uint16
        
        # rotate cell by the major axis       
        #cell = cell.rotate("angel")
        
        fig = Image.fromarray(cell)
        fig = resize_pad(fig)
        name = "%s_cell%s.%s" % (nu[i],str(i), "png")
        name = name.replace("/", "_")
        savepath= os.path.join(imageoutput, name)
        #plt.savefig(savepath)
        fig.save(savepath)
