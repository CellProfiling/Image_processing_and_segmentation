#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 12:52:38 2018
Last Change: 21/05/2018 14:16

@author: trangle

Modified by:
@author: cwinsnes

"""
import click
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
from scipy import ndimage as ndi
from PIL import Image
import gzip
import matplotlib.patches as mpatches

from Segmentation_pipeline_helper import find, watershed_lab, resize_pad
from Segmentation_pipeline_helper import shift_center_mass, pixel_norm

############################################
# EXECUTION PIPELINE FOR CELL SEGMENTATION #
############################################


def extract_img_arrays(microtubule_imgs, protein_imgs, nuclei_imgs):
    """
    Extract the numerical arrays that represent the input images.
    Extracts only the relevant channel of the input image if it is RGB,
    blue for nuclei, red for microtubule, and green for proteins.

    This always results in every returned array being 2D (grayscale).

    The function assumes that all the input lists are of the same length.
    If an image is missing, all channels on the same index will be skipped.

    Arguments:
        microtubule_imgs: A list of paths to microtubule images.
        protein_imgs: A list of paths to protein images.
        nuclei_imgs: A list of paths to nuclei images.

    Returns:
        A generator which yields grayscale image tuples from the input images.
        (grayscale microtubule, grayscale protein, grayscale nuclei).

    Raises:
        IndexError: if the input lists are not the same size.
    """
    image_arrays = []
    num_images = len(nuclei_imgs)
    for index in range(num_images):
        nuclei_img = nuclei_imgs[index]
        microtubule_img = microtubule_imgs[index]
        protein_img = protein_imgs[index]

        current_array = []

        for i, img in enumerate([microtubule_img, protein_img, nuclei_img]):
            try:
                if img.endswith('.gz'):
                    file_handle = gzip.open(img)
                else:
                    file_handle = open(img)
                img_arr = plt.imread(file_handle)
                file_handle.close()

                if len(img_arr.shape) > 2:
                    img_arr = img_arr[:, :, i]
                current_array.append(img_arr)

            except IOError as e:
                logging.error('{}, when reading {}'.format(e, img))

        yield current_array


def plot_boundaries(nucleus_array, regions):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(nucleus_array)
    for region in regions:
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


def cut_bounding_box(im_arrays, plot=False):
    """
    Cut out individual cells from images.
    Each of the arguments to this function should be lists of the same length.
    The elements on the same index in the different arrays are assumed to be
    from the same original image.

    Arguments:
        microtubule_arrays: A list of grayscale microtubule image arrays.
        protein_arrays: A list of grayscale protein image arrays.
        nuclei_arrays: A list of grayscale nuclei image arrays.

    Returns:
        A generator which yields a list of cell arrays for each image
        in succession.
    """
    images = []
    for arrays in im_arrays:
        cells = []
        seeds, num = watershed_lab(arrays[2], rm_border=True)

        regions = skimage.measure.regionprops(seeds)
        if plot:
            plot_boundaries(arrays[2], regions)

        for i, region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox
            mask = seeds[minr:maxr, minc:maxc].astype(np.uint8)
            mask[mask != region.label] = 0
            mask[mask == region.label] = 1

            cell_nuclei = pixel_norm(arrays[2][minr:maxr, minc:maxc] * mask)
            cell_nucleoli = pixel_norm(arrays[1][minr:maxr, minc:maxc] * mask)
            cell_microtubule = np.full_like(cell_nuclei, 0)

            cell = np.dstack((cell_microtubule, cell_nucleoli, cell_nuclei))
            cell = (cell * 255).astype(np.uint8)  # the input file was uint16

            # Align cell orientation
            theta = region.orientation * 180 / np.pi  # radiant to degree
            cell = ndi.rotate(cell, 90 - theta)

            cells.append(cell)
        yield cells

    return images


@click.command()
@click.argument('imageinput')
@click.argument('imageoutput')
@click.option('--blue-suffix', default='blue.tif.gz',
              help='Set the blue image suffix')
@click.option('--green-suffix', default='green.tif.gz',
              help='Set the green image suffix')
@click.option('--red-suffix', default='red.tif.gz',
              help='Set the red image suffix')
@click.option('--plot-boundaries', default=False, is_flag=True)
@click.option('--verbose', default=False, is_flag=True)
def main(imageinput, imageoutput, blue_suffix, green_suffix, red_suffix,
         verbose, plot_boundaries):
    if not os.path.exists(imageoutput):
        os.makedirs(imageoutput)

    if verbose:
        numeric_level = getattr(logging, 'INFO')
        logging.basicConfig(level=numeric_level)

    nuclei_imgs = find(imageinput, suffix=blue_suffix, recursive=False)
    microtubule_imgs = []
    protein_imgs = []
    logging.info('Finding images')

    for nucleus_img in nuclei_imgs:
        microtubule_imgs.append(nucleus_img.replace(blue_suffix, red_suffix))
        protein_imgs.append(nucleus_img.replace(blue_suffix, green_suffix))

    logging.info('Setting up extraction')
    im_arrays = extract_img_arrays(microtubule_imgs, protein_imgs, nuclei_imgs)

    logging.info('Setting up bounding box separation')
    cells = cut_bounding_box(im_arrays, plot=plot_boundaries)

    if plot_boundaries:
        logging.info('Segmentation plots enabled')
    logging.info('Segmenting')
    for i, (image, filename) in enumerate(zip(cells, nuclei_imgs)):
        if verbose:
            progress = i / len(nuclei_imgs) * 100
            print('\r{:.2f}% done'.format(progress), end='', flush=True)

        filename = filename.replace(blue_suffix, '')
        filename = os.path.basename(filename)
        filename = os.path.join(imageoutput, filename)

        for i, cell in enumerate(image):
            fig = resize_pad(cell)
            fig = shift_center_mass(fig)
            fig = Image.fromarray(fig)

            savename = filename + str(i) + '.png'
            fig.save(savename)

    if verbose:
        print()

if __name__ == '__main__':
    main()
