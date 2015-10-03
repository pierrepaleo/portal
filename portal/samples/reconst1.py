#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division
import numpy as np
import portal

import glob
import os


def get_proj_num(fname):
    num_str = fname.split('_')[-1].split('.')[0]
    return num_str

def read_edf(fname, check=False):
    if check:
        if not(os.path.isfile(fname)): raise Exception('File %s not found' % fname)
    return portal.utils.io.edf_read(fname)


def write_edf(fname, data):
    portal.utils.io.edf_write(data, fname) # "a+"





if __name__ == '__main__':


    # - The input folder must contain *sinograms* (not projections)
    # - It should contain files in the format "fileprefix_xxxx.edf" (eg. mysample_scan_sino_xxxx.edf)
    # - All sinograms must be of the same size
    # -----
    folder = '/home/paleo/Projets/ID16/A11_osmium_tomo11/raw_sino' # must contain sinograms, not projections !
    file_prefix = 'sino_'
    do_straighten_sino = 1 # Straighten sinograms to reduce cupping effect
    do_destripe_sino = 0 # Correct rings artifacts by pre-processing the sinogram
    destripe_algorithm = 1 # 1 = Münch, 2 = Titarenko
    slice_start = 201
    slice_end =   210
    sirt_iterations = 300
    folder_out = '/home/paleo/Projets/ID16/A11_osmium_tomo11/rec_portal'
    file_prefix_out = 'rec_'
    # -----












    if not(os.path.isdir(folder)): raise Exception('Not a folder : %s' % folder)
    if not(os.path.isdir(folder_out)): raise Exception('Not a folder : %s' % folder_out)
    pref = os.path.join(folder, file_prefix)

    fl = glob.glob(pref + "????.edf")
    fl.sort()
    if fl == []: raise Exception('could not find any EDF file in %s matching the file prefix %s' % (folder_in, file_prefix))
    n = len(fl)

    if slice_start > n: raise Exception('slice_start (%d) exceeds the number of available slices (%d)' % (slice_start, n))
    if slice_end > n: raise Exception('slice_end (%d) exceeds the number of available slices (%d)' % (slice_end, n))

    sino0 = read_edf(fl[slice_start])
    nAng, npix = sino0.shape
    #~ AST = portal.operators.tomography.AstraToolbox(npix, nAng)
    #~ PT = lambda y : AST.backproj(y, filt=False)

    print("Computing the filter...")
    S = portal.algorithms.sirtfilter.SirtFilter(npix, nAng, sirt_iterations, folder_out)
    #~ thefilter = compute_filter(npix, nAng, sirt_iterations, savedir=folder_out)
    print("Done")

    for i in range(slice_start, slice_end+1):
        print("Processing slice %d" % i)
        sino = read_edf(fl[i])
        if do_straighten_sino:
            print("Straightening...")
            sino = portal.preprocess.sinogram.straighten_sino(sino)
        #~ sino = (sino-sino.min())/(sino.max()-sino.min()) # Does not work along straighten
        if do_destripe_sino:
            print("De-striping...")
            if destripe_algorithm == 1: sino = portal.preprocess.rings.munchetal_filter(sino, 5, 3.6) # 1.6 # TODO : these as a parameter
            elif destripe_algorithm == 2: sino = portal.preprocess.rings.remove_stripe_ti(sino)
            else: raise Exception('Unknown de-striping algorithm')
        print("Convolving...")
        res = S.reconst(sino)
        #~ res = PT(sino) # For dataset A11, "sirt_filter" works well because plain BP is not bad, but FBP enhances the noise too much => need "filter"

        fname_out = os.path.join(folder_out, file_prefix_out) + str("%04d" % i) + ".edf"
        write_edf(fname_out, res)
        print("Wrote %s" % fname_out)

        #~ portal.utils.io.call_imagej([res, res_fbp])



