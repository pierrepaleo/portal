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


from __future__ import division
import numpy as np
from math import sqrt
from portal.image.image import norm2sq
import astra




class AstraToolbox:
    '''
        ASTRA toolbox wrapper.
    '''

    def __init__(self, n_pixels, n_angles):
        '''
        Initialize the ASTRA toolbox with a simple parallel configuration.
        The image is assumed to be square, and the detector count is equal to the number of rows/columns.
        '''

        if isinstance(n_pixels, int):
            n_x, n_y = n_pixels, n_pixels
        else: # assuming iterable
            n_pixels = tuple(n_pixels)
            n_x, n_y = n_pixels
        if isinstance(n_angles, int):
            angles = np.linspace(0, np.pi, n_angles, False)

        self.vol_geom = astra.create_vol_geom(n_x, n_y)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, n_pixels, angles)
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        #~ self.rec_id = astra.data2d.create('-vol', self.vol_geom)

    def backproj(self, sino_data, filt=False, ext=False):
        if filt is True:
            if ext is True:
                bid, rec = astra.create_backprojection(self.filter_projections_ext(sino_data), self.proj_id)
            else:
                bid, rec = astra.create_backprojection(self.filter_projections(sino_data), self.proj_id)
        else:
            bid, rec = astra.create_backprojection(sino_data, self.proj_id)
        astra.data2d.delete(bid)
        return rec

    def proj(self, slice_data):
        sid, proj_data = astra.create_sino(slice_data, self.proj_id)
        astra.data2d.delete(sid)
        return proj_data

    def filter_projections(self, proj_set):
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real


    def filter_projections_ext(self, sino):
        # Extension with boundaries, can be turned into zero-extension.
        n_angles, n_px = sino.shape
        N = nextpow2(2*n_px)
        isodd = 1 if (n_px & 1) else 0
        sino_extended = np.zeros((n_angles, N))

        sino_extended[:, :n_px] = sino
        boundary_right = (sino[:, -1])[:,np.newaxis]
        sino_extended[:, n_px:(n_px+N)/2] = np.tile(boundary_right, (1, (N-n_px)/2))
        boundary_left = (sino[:, 0])[:,np.newaxis]
        sino_extended[:, (n_px+N)/2:N] = np.tile(boundary_left, (1, (N-n_px)/2+isodd))

        ramp = 1./(N/2) * np.hstack((np.arange(N/2), np.arange(N/2, 0, -1)))
        sino_extended_f = np.fft.fft(sino_extended, N, axis=1)

        return np.fft.ifft(ramp * sino_extended_f, axis=1)[:,:n_px].real

    def run_algorithm(self, alg, n_it, data):
        rec_id = astra.data2d.create('-vol', self.vol_geom)
        sino_id = astra.data2d.create('-sino', self.proj_geom, data)
        cfg = astra.astra_dict(alg)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        print("Running %s" %alg)
        astra.algorithm.run(alg_id, n_it)
        rec = astra.data2d.get(rec_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
        return rec

    def cleanup(self):
        #~ astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.proj_id)




class ConvolutionOperator:
    def __init__(self, kernel, initfrom=None):
        if initfrom is None:
            self.kernel = kernel
            self.is2D = True if len(kernel.shape) > 1 else False
            self.mode = 'reflect' #{'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        else :
            self.kernel = np.copy(initfrom.kernel)
            self.is2D = initfrom.is2D
            self.mode = initfrom.mode


    def __mul__(self, img):
        '''
        do the actual convolution.
        If the kernel is 1D, a separable convolution is done.
        '''
        if self.is2D:
            return filters.convolve(img, self.kernel, self.mode)
        else:
            res = filters.convolve1d(img, self.kernel, axis= -1, mode=self.mode)
            return filters.convolve1d(res, self.kernel, axis=0, mode=self.mode)


    def T(self):
        res = ConvolutionOperator(self, initfrom=self)
        res.kern = res.kernel.T
        return res





def power_method(K, Kadj, data, n_it=10):
    '''
    Calculates the norm of operator K
    i.e the sqrt of the largest eigenvalue of K^T*K
        ||K|| = sqrt(lambda_max(K^T*K))

    K : forward operator
    Kadj : backward operator (adjoint of K)
    data : initial data
    '''
    x = Kadj(data)
    for k in range(0, n_it):
        x = Kadj(K(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)



def check_adjoint(K, Kadj, K_input_shape, Kadj_input_shape, tol=1e-7):
    '''
    Checks if the operators K and Kadj are actually adjoint of eachother, i.e if
        < K(x), y > = < x, Kadj(y) >
    '''

    x = np.random.rand(K_input_shape[0], K_input_shape[1])
    y = np.random.rand(Kadj_input_shape[0], Kadj_input_shape[1])

    err = abs(dot(K(x), y) - dot(x, Kadj(y)))
    return False if err > tol else True



















