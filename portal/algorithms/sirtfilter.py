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

'''
    Simplified implementation of https://github.com/dmpelt/pysirtfbp

    An iterative method with L2 regularization (SIRT) amounts to the minimization of
        f(x) = 0.5* ||P*x - d||_2^2
    At each iteration :
        x_{n+1} = x_n - alpha * grad_f (x_n)
                = x_n - alpha * (PT*(P*x - d))      ; PT is the transpose of P
                = (I - alpha*PT*P) x + alpha*PT*d
    This linear recurrence relation leads to :
        x_n = A^n * x_0 + alpha * [sum_{k=0}^{n-1} A^k] P^T * d
        with A = (I - alpha*PT*P)
    This looks like a "backproject-then-filter" method.
    The filter [sum_{k=0}^{n-1} A^k] only depends on the geometry of the dataset,
    so it can be pre-computed for different geometries.
    Once the filter pre-computed, SIRT is equivalent to a simple Filtered Backprojection.
    The filtering is done in the sinogram domain.

'''


from __future__ import division
import numpy as np
from portal.operators.tomography import AstraToolbox



class SirtFilter:
    def __init__(self, n_pixels, angles, n_it, savedir=None):
        '''
        Initialize the SIRT-Filter class.

        n_pixels : number of detector pixels.
            Can be either an integer (in that case, the slice is square), either a tuple (n_x, n_y).
        n_angles : projection angles.
            Can be either an integer (in that case, the angles are even-spaced between 0 and pi), either a numpy array containing custom angles.
        n_it : number of iterations for the SIRT algorithm.
        '''

        # Initialize ASTRA with the provided geometry
        # FIXME : only checked for square slices
        if not(isinstance(n_pixels, int)):
            if n_pixels[0] != n_pixels[1]: raise Exception('SirtFilter only works with square slices')
            self.n_px = n_pixels[0]
        else: self.n_px = n_pixels
        if not(isinstance(angles, int)):
            n_a = len(tuple(angles))
        else: n_a = angles

        self.AST = AstraToolbox(n_pixels, angles)
        self.nit = n_it
        self.thefilter = _compute_filter(savedir)



    def _compute_filter_operator(self, npix, P, PT, alph, n_it=20, lambda_tikhonov=0):
        x = np.zeros((npix,npix),dtype=np.float32)
        x[npix//2,npix//2]=1
        #~ xs = np.copy(x)
        xs = np.zeros_like(x)
        for i in range(n_it):
            xs += x
            x -= alph*PT(P(x)) + alph*lambda_tikhonov
            #~ astra.extrautils.clipCircle(x)
        return xs



    def _compute_filter(self, savedir=None):

        npix = self.n_px
        nAng = self.n_A
        niter = self.n_it

        # Check if filter is already calculated for this geometry
        if savedir is not None:
            if not(os.path.isdir(savedir)): raise Exception('%s no such directory' % savedir)
            fname = str_implode(['sirt_filter', str(npix), str(nAng), str(niter)], '_') + '.npz'
            fname = os.path.join(savedir, fname)
            if os.path.isfile(fname):
                nz_desc = np.load(fname)
                nz_data = nz_desc['data']
                nz_geom = nz_desc['geometry']
                nz_iter = nz_desc['iterations']
                nz_desc.close()
                if nz_geom[0] != npix or nz_geom[1] != nAng:
                    print('Warning : file %s does not match the required geometry. Re-computing the filter' % fname)
                elif nz_iter != niter:
                    print('Warning : file %s does not seem to have the correct number of iterations. Re-computing the filter' % fname)
                else:
                    print('Loaded %s' % fname)
                    return nz_data
            else:
                print('Filter %s not found. Computing the filter.' % fname)

        nDet = npix
        alph = 1./(nAng*nDet)

        # Always use an odd number of detectors (to be able to set the center pixel to one)
        if npix % 2 == 0: npix += 1

        # Initialize ASTRA with this new geometry
        AST = AstraToolbox(npix, nAng)
        P = lambda x : self.AST.proj(x) #*3.14159/2.0/nAng
        PT = lambda y : self.AST.backproj(y, filt=False)

        # Compute the filter with this odd shape
        xs = _compute_filter_operator(npix, P, PT, alph, niter)

        # The filtering is done in the sinogram domain, using FFT
        # The filter has to be forward projected, then FT'd

        # Forward project
        ff = alph*P(xs)

        # The convolution theorem states that the size of the FFT should be
        # at least 2*N-1  where N is the original size.
        # Here we make both slice (horizontal size N in real domain) and filter (possibly N+1 in real domain)
        # have a size nextpow2(N)*2 in Fourier domain, which should provide good performances for FFT.
        nexpow = _ceilpow2(nDet)

        # Manual fftshift
        f = np.zeros((nAng, 2*nexpow+0*1),dtype=np.float32) # <---
        # Half right (+1) goes to the left
        f[:, 0:int(npix / 2) + 1] = ff[:, int(npix / 2):npix]
        # Half left goes to the right
        f[:, f.shape[1] - int(npix / 2):f.shape[1]] = f[:, 1:int(npix / 2) + 1][:, ::-1]
        # FFT
        f_fft = np.fft.fft(f,axis=1)

        # Result should be real, since it will be multiplied and backprojected.
        # With the manual zero-padding, the filter is real and symmetric, so its Fourier
        # Transform is also real and symmetric.
        result = f_fft.real

        if savedir is not None:
            np.savez_compressed(fname, data=result, geometry=np.array([nDet, nAng]), iterations=niter)
        return result

    def _convolve(self, sino, thefilter):
        npx = sino.shape[1]
        sino_f = np.fft.fft(sino, 2*npx + 0*1, axis=1) * thefilter # FIXME : consider nextpow2(npx)*2 for speed
        return np.fft.ifft(sino_f , axis=1)[:, :npx].real


    def reconst(self, sino):
        s = _convolve(sino, self.thefilter)
        return self.AST.backproj(s, filt=False)



