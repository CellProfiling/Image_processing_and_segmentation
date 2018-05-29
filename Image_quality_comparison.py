#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:04:13 2018

@author: trangle
"""
#########################
### Function
#########################
import sys
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    #img1 = normalize(img1)
    #img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)



#########################
### Pixel difference
#########################
# Manhattan norm = the sum of the absolute values (delta) = how much the images differ
# Zero norm = number of delta equals to zero = how many pixels differs

im1 = plt.imread('gen_PNG_numt_418_A1_3__cell1.png')
im1_ori = plt.imread('418_A1_3__cell1.png')

im2 = plt.imread('gen_PNG_numt_528_G1_3__cell1.png')
im2_ori = plt.imread('528_G1_3__cell1.png')

channel = ["microtubule channel", "nucleoli channel","nuclei channel"]
for c in range(0,im1.shape[2]):
    n_m, n_z = compare_images(im1[:,:,c],im1_ori[:,:,c])
    print(channel[c])
    #print('Manhattan norm: {0} / per pixel: {1}'.format(n_m,round(n_m/im1[:,:,c].size),4)))
    print("Manhattan norm:", n_m, "/ per pixel:", n_m/im1[:,:,c].size)
    print("Zero norm:", n_z, "/ per pixel:", n_z*1.0/im1[:,:,c].size)
    

#########################
### NRMSE and SSIM
#########################
# Normalized root mean squared error, very similar to Manhattan norm
# Structural SIMilarity (SSIM) index (https://ece.uwaterloo.ca/~z70wang/research/ssim/)
import numpy
from scipy.signal import fftconvolve

def ssim(im1, im2, window, k=(0.01, 0.03), l=255):
    """See https://ece.uwaterloo.ca/~z70wang/research/ssim/"""
    # Check if the window is smaller than the images.
    for a, b in zip(window.shape, im1.shape):
        if a > b:
            return None, None
    # Values in k must be positive according to the base implementation.
    for ki in k:
        if ki < 0:
            return None, None
    c1 = (k[0] * l) ** 2
    c2 = (k[1] * l) ** 2
    window = window/numpy.sum(window)
    
    mu1 = fftconvolve(im1, window, mode='valid')
    mu2 = fftconvolve(im2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = fftconvolve(im1 * im1, window, mode='valid') - mu1_sq
    sigma2_sq = fftconvolve(im2 * im2, window, mode='valid') - mu2_sq
    sigma12 = fftconvolve(im1 * im2, window, mode='valid') - mu1_mu2
    
    if c1 > 0 and c2 > 0:
        num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den
    else:
        num1 = 2 * mu1_mu2 + c1
        num2 = 2 * sigma12 + c2
        den1 = mu1_sq + mu2_sq + c1
        den2 = sigma1_sq + sigma2_sq + c2
        ssim_map = numpy.ones(numpy.shape(mu1))
        index = (den1 * den2) > 0
        ssim_map[index] = (num1[index] * num2[index]) / (den1[index] * den2[index])
        index = (den1 != 0) & (den2 == 0)
        ssim_map[index] = num1[index] / den1[index]
        
    mssim = ssim_map.mean()
    return mssim, ssim_map


def nrmse(im1, im2):
    a, b = im1.shape
    rmse = numpy.sqrt(numpy.sum((im2 - im1) ** 2) / float(a * b))
    max_val = max(numpy.max(im1), numpy.max(im2))
    min_val = min(numpy.min(im1), numpy.min(im2))
    return (rmse / (max_val - min_val))

for c in range(0,im1.shape[2]):
    if __name__ == "__main__":
        import sys
        from scipy.signal import gaussian
        from PIL import Image
    
        img1 = Image.fromarray(im1[:,:,c])#Image.open(sys.argv[1])
        img2 = Image.fromarray(im1_ori[:,:,c])#Image.open(sys.argv[2])
    
        if img1.size != img2.size:
            print("Error: images size differ")
            raise SystemExit
    
        # Create a 2d gaussian for the window parameter
        win = numpy.array([gaussian(11, 1.5)])
        win2d = win * (win.T)
    
        num_metrics = 2
        sim_index = [2 for _ in range(num_metrics)]
        for band1, band2 in zip(img1.split(), img2.split()):
            b1 = numpy.asarray(band1, dtype=numpy.double)
            b2 = numpy.asarray(band2, dtype=numpy.double)
            # SSIM
            res, smap = ssim(b1, b2, win2d)
    
            m = [res, nrmse(b1, b2)]
            for i in range(num_metrics):
                sim_index[i] = min(m[i], sim_index[i])
    
        print(channel[c], "[nrmse, ssim] =", sim_index)
    
#Segmentation    
nu = im2[:,:,2]

    '''if segment == True:
        nu = original[:,:,2]
        marker = np.full_like(nu,0)
        #marker[nu > threshold_otsu(nu)] = 2
        marker[nu > 0] = 2
        marker[nu == 0] = 1
        
        elevation_map = sobel(nu)
        segmentation = watershed(elevation_map, marker)
        segmentation = ndi.binary_fill_holes(segmentation - 1)
        
        labeled_obj, num = ndi.label(skimage.morphology.opening(segmentation))
        
        #snake = active_contour(gaussian(nu, 3),init, alpha=0.015, beta=10, gamma=0.001)

        for obj in skimage.measure.regionprops(labeled_obj):
            minr, minc, maxr, maxc = obj.bbox
            img1 = Image.fromarray(original[:,:,1][minr:maxr,minc:maxc]*255)
            img2 = Image.fromarray(generated[:,:,1][minr:maxr,minc:maxc]*255)
    else:
        img1 = Image.fromarray(original[:,:,1])
        img2 = Image.fromarray(generated[:,:,1])
    
    f, axarr = plt.subplots(1,2)
    axarr[0] = plt.imshow(img1)
    axarr[1] = plt.imshow(img2)
    '''   
def compare_img(original,generated,segment=True):
    # Create a 2d gaussian for the window parameter
    img1=original
    img2=generated
    win = numpy.array([gaussian(11, 1.5)])
    win2d = win * (win.T)
    
    num_metrics = 2
    sim_index = [2 for _ in range(num_metrics)]
    for band1, band2 in zip(img1.split(), img2.split()):
        b1 = numpy.asarray(band1, dtype=numpy.double)
        b2 = numpy.asarray(band2, dtype=numpy.double)
        # SSIM
        res, smap = ssim(b1, b2, win2d)
        m = [res, nrmse(b1, b2)]
        for i in range(num_metrics):
            sim_index[i] = min(m[i], sim_index[i])
    
    return sim_index

a = compare_img(im1_ori,im1)
print("Channel nucleoli: [nrmse, ssim] =", a)

#################
import sys
from scipy.signal import gaussian
from PIL import Image
import skimage
import pandas as pd
import pickle 
gen_path = '/Users/ngoc.le/Desktop/gen_test_images'
gen = find(gen_path,prefix='gen_', suffix=".png",recursive=False) #blue chanel =nu

ori = []
for f in gen:
    f=f.replace('gen_test_images','nucleoli_v18_MCF7_noccd/MCF7_nuclei')
    f=f.replace('gen_','')
    ori.append(f)

results =  pd.DataFrame(index=range(0,len(gen)),
                        columns=['image', 'NRMSE', 'SSIM','Pearson_nu','p_nu','Pearson_no','p_no','Pearson_mt','p_mt'])
for index,imgpath in enumerate(gen):
    
    print("Comparing image {0}/{1} to original".format(index+1,len(gen)))
    name0=(gen[index].split("images/"))[1].split("blue")[0]
    # Unzip .gz file and read content image to img
    try:
        im = plt.imread(imgpath)
        #im = im[:,:,1]
    except:
        print("%s does not have valid nucleoli channel" % name0)
        continue
    
    # Unzip .gz file and read content image to img
    try:
        im_ori = plt.imread(ori[index])
        #im_ori = im_ori[:,:,1]
    except:
        print("%s does not have valid nucleoli channel" % name0)
        continue

    nu = im_ori[:,:,2]
    marker = np.full_like(nu,0)
    #marker[nu > threshold_otsu(nu)] = 2
    marker[nu > 0] = 2
    marker[nu == 0] = 1
        
    elevation_map = sobel(nu)
    segmentation = watershed(elevation_map, marker)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
        
    labeled_obj, num = ndi.label(skimage.morphology.opening(segmentation))        
    #snake = active_contour(gaussian(nu, 3),init, alpha=0.015, beta=10, gamma=0.001)
    for obj in skimage.measure.regionprops(labeled_obj):
        if obj.area >= 2000:
            minr, minc, maxr, maxc = obj.bbox
            img1_no = im_ori[:,:,1][minr:maxr,minc:maxc]
            img2_no = im[:,:,1][minr:maxr,minc:maxc]
            img1_nu = im_ori[:,:,2][minr:maxr,minc:maxc] #nucleichannel
            img2_nu = im[:,:,2][minr:maxr,minc:maxc] #nucleichannel
            img1_mt = im_ori[:,:,0][minr:maxr,minc:maxc] #microtubule channel
            img2_mt = im[:,:,0][minr:maxr,minc:maxc] #nmicrotubule channel
            
    #results = results.append([gen[index].split('images/')[1], a], ignore_index = True)
    results.image[index] = name0
    
    
    results.NRMSE[index] = nrmse(img1_no,img2_no) #NRMSE
    results.SSIM[index] = ssim(img2_no,img1_no) #SSIM
    
    pearson=pearsonr(img1_nu.flatten(),img2_nu.flatten())
    results.Pearson_nu[index] = pearson[0]
    results.p_nu[index] = pearson[1]
    pearson=pearsonr(img1_no.flatten(),img2_no.flatten())
    results.Pearson_no[index] = pearson[0]
    results.p_no[index] = pearson[1]
    pearson=pearsonr(img1_mt.flatten(),img2_mt.flatten())
    results.Pearson_mt[index] = pearson[0]
    results.p_mt[index] = pearson[1]
    
PC3_nuclei = results
PC3_nuclei.to_csv('/Users/ngoc.le/Desktop/PC3_latent32_1_MSE_epoch500.csv')

MCF7_nuclei = results
MCF7_nuclei.SSIM

MCF7_nuclei.to_csv('/Users/ngoc.le/Desktop/MCF7_latent64_1_MSE.csv')

U2OS_numt = results
U2OS_numt.to_csv('/Users/ngoc.le/Desktop/U2OS_numt_latent16_3_MSE.csv')

a = np.hstack((MCF7_nuclei.SSIM.normal(size=1000),
               MCF7_nuclei.SSIM.normal(loc=5, scale=2, size=1000)))
