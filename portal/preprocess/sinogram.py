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
from portal.algorithms.simplex import _minimize_neldermead
from math import pi

# ------------------------------------------------------------------------------
# ------------Cupping reduction : "sinogram straightening" ---------------------
# ------------------------------------------------------------------------------



def straighten_sino(sino, order=3):
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
        z = np.polyfit(x, y, order)
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
    radius = n_px/8. #n_px #small radius => big complement of double-wedge
    s_vec = np.arange(smin, smax+1, sstep)*1.0
    Q_vec = np.zeros_like(s_vec)
    for i,s in enumerate(s_vec):
        print("[calc_center_shifts] Case %d/%d" % (i+1,s_vec.shape[0]))
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

    #figure(); imshow(sino2);

        # Create the mask "outside double wedge" (see [1])
        R, C = generate_coords(sino2.shape)
        mask = 1 - (np.abs(R) <= np.abs(C)*radius)

        # Take FT of the sinogram and compute the Fourier metric
        sino_f = np.abs(np.fft.fftshift(np.fft.fft2(sino2)))

    #figure(); imshow(np.log(1+sino_f) * mask, interpolation="nearest"); colorbar();

        #~ sino_f = np.log(sino_f)
        Q_vec[i] = np.sum(sino_f * mask)/np.sum(mask)

    s0 = s_vec[Q_vec.argmin()]
    return n_px/2 + s0/2 - 0.5






def centroid_objective(X, n_angles, centr):
    """
    Helper function for get_center()
    """
    offs, amp, phi = X
    t = np.linspace(0, n_angles, n_angles)
    _sin = offs + amp * np.sin(2*pi*(1./(2*n_angles))*t + phi)
    return np.sum((_sin - centr)**2)





def get_center(sino, debug=False):
    '''
    Determines the center of rotation of a sinogram by computing the center of gravity of each row.
    The array of centers of gravity is fitted to a sine function.
    The axis of symmetry is the estimated center of rotation.
    '''

    n_a, n_d = sino.shape
    # Compute the vector of centroids of the sinogram
    i = range(n_d)
    centroids = np.sum(sino*i, axis=1)/np.sum(sino, axis=1)

    # Fit with a sine function : phase, amplitude, offset.
    # Uses Nelder-Mead downhill-simplex algorithm
    cmax, cmin = centroids.max(), centroids.min()
    offs = (cmax + cmin)/2.
    amp = (cmax - cmin)/2.
    phi = 1.1 # !
    x0 = (offs, amp, phi)
    sol, _energy, _iterations, _success, _msg = _minimize_neldermead(centroid_objective, x0, args=(n_a, centroids))

    offs, amp, phi = sol

    if debug:
        t = np.linspace(0, n_a, n_a)
        _sin = offs + amp * np.sin(2*pi*(1./(2*n_a))*t + phi)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(centroids); plt.plot(_sin)
        plt.show()

    return offs


