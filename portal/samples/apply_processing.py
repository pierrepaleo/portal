#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This example shows the usage of apply_processing() and explore_tomo_dataset() functions.
The input folder should contain *sinogram* (not radios !) in EDF format.
"""

import numpy as np
import portal
import os

def myprocessing(sino, args):

    S = args

    print("Scaling...")
    sino = 40.*(sino-sino.min())/(sino.max()-sino.min())
    print("Straightening...")
    sino = portal.preprocess.sinogram.straighten_sino(sino, order=2)
    print("De-striping...")
    sino = portal.preprocess.rings.munchetal_filter(sino, 5, 3.6)
    print("Reconstructing...")
    res = S.reconst(sino)
    portal.operators.tomography.clipCircle(res) # optional...
    return res


if __name__ == '__main__':

    # Parameters for input dataset
    folder = 'sino'
    file_prefix = 'sino_'
    slice_width = 2048
    num_projections = 1500
    rot_center = 1024 + 38 # ...

    # Parameters for output dataset
    slice_start = 150
    slice_end =   150
    folder_out = 'rec_sirtfilter'
    file_prefix_out = 'rec_'

    # Parameters for the reconstruction
    sirt_iterations = 300

    # Create the SIRT-Filter which shall be used in the processing of all sinograms
    S = portal.algorithms.sirtfilter.SirtFilter(slice_width, num_projections, sirt_iterations, savedir=folder_out, rot_center=rot_center)

    # apply the custom processing routine on the dataset
    dataset = portal.utils.esrf.explore_tomo_dataset(folder, file_prefix)

    options = {}
    options['start'] = slice_start
    options['end'] = slice_end
    options['verbose'] = 1
    options['output_folder'] = 'rec/'
    options['output_file_prefix'] = 'rec_'
    portal.utils.esrf.apply_processing(myprocessing, dataset, options, extra_args=S)

