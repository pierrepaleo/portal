#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of ESRF nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import numpy as np
from portal.utils.misc import generate_coords

__all__ = ['center_sino', 'straighten_sino']

# ------------------------------------------------------------------------------
# ------------Cupping reduction : "sinogram straightening" ---------------------
# ------------------------------------------------------------------------------



def straighten_sino(sino):
    n_angles, n_pix = sino.shape
    x = np.arange(n_pix)
    sino_corr = np.zeros_like(sino)

    i = 0
    for line in range(n_angles):
        y = sino[line, :]
        # Least-Squares, 3rd order polynomial :
        #~ X = np.array([np.ones(n_pix), x, x**2, x**3]).T
        #~ X = X[:, ::-1] # numpy convention
        #~ z0 = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        z = np.polyfit(x, y, 3)
        f = np.poly1d(z)
        sino_corr[line, :] = y - f(x)
    return sino_corr





# ------------------------------------------------------------------------------
# ------------ Determine the center of rotation --------------------------------
# ------------------------------------------------------------------------------

def _gaussian_kernel(ksize, sigma):
    '''
    Creates a gaussian function of length "ksize" with std "sigma"
    '''
    x = np.arange(ksize) - (ksize - 1.0) / 2.0
    gaussian = np.exp(-(x / sigma) ** 2 / 2.0).astype(np.float32)
    #gaussian /= gaussian.sum(dtype=np.float32)
    return gaussian


# TODO : test on other data
def calc_center_shifts(sino, smin, smax, sstep=1):
    '''
    Determines the center of rotation according to
        Vo NT Et Al, "Reliable method for calculating the center of rotation in parallel-beam tomography", Opt Express, 2014

    The idea is the following :
        - create a reflected version of the original sinogram
        - shift this mirrored sinogram and append it to the original one
        - take the Fourier Transform and see what happens in the vertical line (u = 0)
        - repeat for different shifts

    sino : sinogram as a numpy array
    smin: minimum shift of lower sinogram (can be negative)
    smax: maximum shift of lower sinogram
    sstep: shift step (can be less than 1 for subpixel precision)
    '''

    if sstep < 1: raise NotImplementedError('subpixel precision is not implemented yet...')
    sino_flip = sino[::-1, :]
    n_angles, n_px = sino.shape
    radius = n_px/2 #small radius => big complement of double-wedge
    s_vec = np.arange(smin, smax+1, sstep)*1.0
    Q_vec = np.zeros_like(s_vec)
    for i,s in enumerate(s_vec):
        #~ if _VERBOSE: print("[calc_center_shifts] Case %d/%d" % (i+1,s_vec.shape[0]))
        # Create the artificial 360° sinogram (cropped)
        sino2 = np.zeros((2*n_angles, n_px - abs(s)))
        if s > 0:
            sino2[:n_angles, :] = sino[:, s:]
            sino2[n_angles:, :] = sino_flip[:, :-s]
        elif s < 0:
            sino2[:n_angles, :] = sino[:, :s]
            sino2[n_angles:, :] = sino_flip[:, -s:]
        else:
            sino2[:n_angles, :] = sino
            sino2[n_angles:, :] = sino_flip

        # Create the mask "outside double wedge" (see [1])
        R, C = generate_coords(sino2.shape)
        mask = 1 - (np.abs(R) <= np.abs(C)*radius)

        # Take FT of the sinogram and compute the Fourier metric
        sino_f = np.abs(np.fft.fftshift(np.fft.fft2(sino2)))
        #~ sino_f = np.log(sino_f)
        Q_vec[i] = np.sum(sino_f * mask)/np.sum(mask)

    s0 = s_vec[Q_vec.argmin()]
    return n_px/2 + s0




def calc_center_centroids(sino):
    '''
    Determines the center of rotation of a sinogram by computing the center of gravity of each row.
    The array of centers of gravity is fitted to a sine function.
    The axis of symmetry is the estimated center of rotation.
    '''

    n_angles, n_px = sino.shape
    # Compute the center of gravity for each row
    i = np.arange(n_px)
    centroids = np.sum(sino*i, axis=1)/np.sum(sino, axis=1) # sino*i : each row is mult. by "i"
    # Fit this array to a sine function.
    # This is done by taking the Fourier Transform and keeping the first two components (mean value and fund. frequency)
    centroids_f = np.fft.fft(centroids)
    #~ sigma = 9.0 # blur factor
    #~ Filter = np.fft.ifftshift(_gaussian_kernel(n_angles, sigma))
    Filter = np.zeros(n_angles); Filter[0:2] = 1; Filter[-2:] = 1
    centroids_filtered = np.fft.ifft(centroids_f * Filter).real

    return centroids_filtered.min()




# ------------------------------------------------------------------------------
#  Trick to customize the center of rotation in ASTRA.
#  This should not be necessary in future versions of ASTRA.
#  Taken from tomopy :
#
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
# ------------------------------------------------------------------------------


def center_sino(sino0, center):
    sino = np.copy(sino0)
    _, ndet = sino.shape
    shft = int(np.round(ndet / 2. - center))
    sino[:] = np.roll(sino, shft)
    l = shft
    r = sino.shape[1] + shft
    if l < 0:
        l = 0
    if r > sino.shape[1]:
        r = sino.shape[1]
    sino[:, 0:l] = 0
    sino[:, r:sino.shape[1]] = 0
    return sino
















