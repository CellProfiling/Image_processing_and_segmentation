#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:18:24 2018

@author: trang.le
"""
# transforming png files [height,width,channel] to h5 format [channel,stack,height,weight]
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import Segmentation_pipeline_helper

imageinput = "~/U2OS_noccd/PNG"
imageoutput = "~/U2OS_noccd/h5"

if not os.path.exists(imageoutput):
    os.makedirs(imageoutput)
    
l = find(imageinput,prefix=None, suffix=".png",recursive=False)
savepath=[]
for filepath in l:
    filepath=filepath.replace('.png','.h5')
    filepath=filepath.replace('PNG','h5')
    savepath.append(filepath)
    
for index,imgpath in enumerate(l):
    im = plt.imread(imgpath)
    x= np.transpose(im,(2,0,1))
    x= np.expand_dims(x, axis=1)
    h = h5py.File(savepath[index],"w")
    h.create_dataset('image',data=x)
    h.close
