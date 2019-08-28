#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 12:52:38 2018

@author: trangle

The original 2048x2048 images were taken with 63x, corresponding to 164.02x164.02x um
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import skimage
import imageio
import pandas as pd
from scipy import ndimage as ndi
import gzip
from Segmentation_pipeline_helper import find, watershed_lab, watershed_lab2, resize_pad, find_border, pixel_norm, shift_center_mass

##### Parse arguments
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--imgInput', type=str, default='/home/trangle/Desktop/cell-cycle-model/data/Cyclin_B_Cyclin_E_mt_chicken3µg', help='Define path to image input directory')
#imageinput = "/afs/pdc.kth.se/projects/cellprofiling/projects/integrated_cell/data/nucleoli_data/nucleoli_images/nucleoli_v18_A-431"
parser.add_argument('--imgOutput', type=str, default='/home/trangle/Desktop/cell-cycle-model/data/segmented_cell', help='Define desired path to save the segmented images')
##### EXECUTION PIPELINE FOR CELL SEGMENTATION
parser.add_argument('--DAPIChannel', type=str, default='_ch01.tif', help='Keyword to find nuclei channel')
parser.add_argument('--MtChannel', type=str, default='_ch02.tif', help='Keyword to find microtubule channel')
parser.add_argument('--OrgChannel', type=str, default='_ch00.tif', help='Keyword to find protein-of-interest channel')

args = parser.parse_args()

if not os.path.exists(args.imgOutput):
    os.makedirs(args.imgOutput)
    
# Make a list of path to the tif.gz files
nuclei = find(args.imgInput,prefix=None, suffix=args.DAPIChannel,recursive=False) #blue chanel =nu
protein = []
microtubule = []
for f in nuclei:
    f=f.replace(args.DAPIChannel,args.OrgChannel)
    protein.append(f)
    f=f.replace(args.OrgChannel,args.MtChannel)
    microtubule.append(f)

# For each image, import 3 chanels
# Use nuclei as seed, microtubule as edges to segment the image
# Cut the bounding box of each cell (3channels) in the respective image, slack and save

def open_image(filename,ch_number):
    if '.gz' in filename:
        with gzip.open(filename) as f:
            ch = imageio.imread(f) #cv2.imread(f)
    else:
        ch = imageio.imread(filename)
            
    if len(ch.shape) > 2:
        ch=ch[:,:,ch_number]
    return ch


data_info = pd.DataFrame()
for index,imgpath in enumerate(nuclei):
    
    name0 = os.path.basename(nuclei[index]).split(args.DAPIChannel)[0]
    print("Segmenting image {0}/{1} : %s".format(index+1,len(protein)) % name0)

    # Unzip .gz file and read content image to img
    
    try:
        nu = open_image(nuclei[index],2)
    except:
        print("%s does not have valid nucleus channel" % name0)
        continue
    
    try:
        org = open_image(protein[index],1)
    except:
        print("%s does not have valid protein channel" % name0)
        continue  
    
    try:
        mi = open_image(microtubule[index],0)
    except:
        print("%s does not have valid microtubule channel" % name0)
        continue      

    # obtain nuclei seed for watershed segmentation
    seed, num = watershed_lab(nu, rm_border=False)
    
    # segment microtubule image
    #ret, thresh = cv2.threshold(mi,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    #kernel = np.ones((3,3),np.uint8)
    #marker = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    
    marker = np.zeros_like(mi)
    marker[mi == 0] = 1 #background
    #marker = skimage.morphology.binary_erosion(marker,skimage.morphology.square(3)).astype(int)
    marker[seed > 0] = seed[seed > 0] + 1 #foreground
    labels = watershed_lab2(mi, marker = marker)
    
    #remove all cells where nucleus is touching the border
    labels = labels - 1
    border_indice = find_border(seed)
    mask = np.in1d(labels,border_indice).reshape(labels.shape)
    labels[mask] = 0
    
    nu_area=[]
    for region in skimage.measure.regionprops(seed):
        nu_area.append([region.label, region.area, region.eccentricity]) 
    #Eccentricity is how much the coinic section (nuclei ) varies from being circular. 0=perfect circle, 1 = 
    nu_area = pd.DataFrame(nu_area, columns = ['Cell_number', 'Nucleus_area', 'Nucleus_roundness'])
    
    '''
    #Plotthing the segmentation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(mi)
    for region in skimage.measure.regionprops(labels):
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
    '''
    # Cut boundingbox and extract information
    cell_info=[]
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
        # filling small holes in the mask
        mask = skimage.morphology.dilation(mask)

        cell_nuclei = nu[minr:maxr,minc:maxc]*mask#pixel_norm(nu[minr:maxr,minc:maxc]*mask)
        cell_org = org[minr:maxr,minc:maxc]*mask
        #cell_org = pixel_norm(org[minr:maxr,minc:maxc]*mask)
        cell_microtubule = mi[minr:maxr,minc:maxc]*mask#pixel_norm(mi[minr:maxr,minc:maxc]*mask)

        # stack channels
        cell = np.dstack((cell_microtubule,cell_org,cell_nuclei))         
        #cell = (cell*255).astype(np.uint8) #the input file was uint16         

        # align cell to the 1st major axis  
        theta=region.orientation*180/np.pi #radiant to degree conversion
        cell = ndi.rotate(cell, 90-theta)

        # center to the center of mass of the nucleus
        fig = shift_center_mass(cell)
        # resize images
        fig = resize_pad(fig) # default size is 256x256
        if fig.max()==0:
            print('segmented cell with no signal in image %s' % name0)
            continue
            print('Segmented image has no signal!')
            
        name = "%s_cell%s.%s" % (name0,str(i), "png")
        name = name.replace("/", "_")
        
        savepath= os.path.join(args.imgOutput, name)
        skimage.io.imsave(savepath, fig)
        #fig.save(savepath)
        
        mask_coord = np.where(mask==1)
        cyclinb1_cellarea = cell_org[mask_coord[0], mask_coord[1]]
        cell_info.append([savepath, region.label, region.area, region.bbox_area, region.eccentricity, cell_org.max(), np.sum(cyclinb1_cellarea), np.mean(cyclinb1_cellarea), np.median(cyclinb1_cellarea)])
    
    cell_info = pd.DataFrame(cell_info, columns = ['Cell_path','Cell_number', 'Cell_area', 'Cell_bb_area','Cell_roundness', 'CyclinB1_max','CyclinB1_sum', 'CyclinB1_avg_nuclei', 'CyclinB1_avg_cell'])
    cell_info = cell_info.merge(nu_area, on='Cell_number')
    
    data_info = data_info.append(cell_info, ignore_index=True)
    
    
data_info.to_csv(os.path.join(args.imgOutput, 'Metadata.csv'))
