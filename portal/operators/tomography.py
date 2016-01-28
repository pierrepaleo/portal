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
from portal.operators.image import norm2sq
#~ from scipy.ndimage import convolve1d
import astra


def convolve1d(t1, t2):
    #~ N = max(t1.shape[1], t2.shape[0])
    #~ t1_f = np.fft.fft(t1, 2*N, axis=1)
    #~ t2_f = np.abs(np.fft.fft(t2, 2*N)) # to avoid manual fftshift
    #~ return np.fft.ifft(t1_f * t2_f, axis=1)[:, :t1.shape[1]].real
    res = np.zeros_like(t1)
    for i in range(t1.shape[0]):
        res[i, :] = np.convolve(t1[i, :], t2,"same")
    return res



def nextpow2(N):
    p = 1
    while p < N:
        p *= 2
    return p


class AstraToolbox:
    """
    ASTRA toolbox wrapper for parallel beam geometry.
    """

    def __init__(self, n_pixels, angles, rot_center=None, fullscan=False, super_sampling=None):
        """
        Initialize the ASTRA toolbox with a simple parallel configuration.
        The image is assumed to be square, and the detector count is equal to the number of rows/columns.

        n_pixels: integer
            number of pixels of one dimension of the image
        angles : integer or numpy.ndarray
            number of projection angles (if integer), or custom series of angles
        rot_center : float
            user-defined rotation center
        fullscan : boolean
            if True, use a 360 scan configuration
        super_sampling : integer
            Detector and Pixel supersampling
        """

        if isinstance(n_pixels, int):
            n_x, n_y = n_pixels, n_pixels
        else: # assuming iterable
            n_pixels = tuple(n_pixels)
            n_x, n_y = n_pixels
        angle_max = np.pi
        if fullscan: angle_max *= 2
        if isinstance(angles, int):
            angles = np.linspace(0, angle_max, angles, False)
        n_angles = angles.shape[0]

        self.vol_geom = astra.create_vol_geom(n_x, n_y)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, n_pixels, angles)

        if rot_center:
            o_angles = np.ones(n_angles) if isinstance(n_angles, int) else np.ones_like(n_angles)
            self.proj_geom['option'] = {'ExtraDetectorOffset': (rot_center - n_x / 2.) * o_angles}
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)

        # vg : Volume geometry
        self.vg = astra.projector.volume_geometry(self.proj_id)
        # pg : projection geometry
        self.pg = astra.projector.projection_geometry(self.proj_id)

        # ---- Configure Projector ------
        # sinogram shape
        self.sshape = astra.functions.geom_size(self.pg)
        # Configure projector
        self.cfg_proj = astra.creators.astra_dict('FP_CUDA')
        self.cfg_proj['ProjectorId'] = self.proj_id
        if super_sampling:
            self.cfg_proj['option'] = {'DetectorSuperSampling':super_sampling}

        # ---- Configure Backprojector ------
        # volume shape
        self.vshape = astra.functions.geom_size(self.vg)
        # Configure backprojector
        self.cfg_backproj = astra.creators.astra_dict('BP_CUDA')
        self.cfg_backproj['ProjectorId'] = self.proj_id
        if super_sampling:
            self.cfg_backproj['option'] = {'PixelSuperSampling':super_sampling}
        # -------------------
        self.rot_center = rot_center if rot_center else n_pixels/2
        self.n_pixels = n_pixels
        self.angles = angles




    def __checkArray(self, arr):
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.flags['C_CONTIGUOUS']==False:
            arr = np.ascontiguousarray(arr)
        return arr


    def backproj(self, s, filt=False, ext=False, method=1):
        if filt is True:
            if ext is True:
                sino = self.filter_projections_ext(s, method)
            else:
                sino = self.filter_projections(s)
        else:
            sino = s
        sino = self.__checkArray(sino)

        # In
        sid = astra.data2d.link('-sino', self.pg, sino)
        self.cfg_backproj['ProjectionDataId'] = sid
        # Out
        v = np.zeros(self.vshape, dtype=np.float32)
        vid = astra.data2d.link('-vol', self.vg, v)
        self.cfg_backproj['ReconstructionDataId'] = vid

        bp_id = astra.algorithm.create(self.cfg_backproj)
        astra.algorithm.run(bp_id)
        astra.algorithm.delete(bp_id)
        astra.data2d.delete([sid, vid])
        return v



    def proj(self, v):
        v = self.__checkArray(v)
        # In
        vid = astra.data2d.link('-vol', self.vg, v)
        self.cfg_proj['VolumeDataId'] = vid
        # Out
        s = np.zeros(self.sshape, dtype=np.float32)
        sid = astra.data2d.link('-sino',self.pg, s)
        self.cfg_proj['ProjectionDataId'] = sid

        fp_id = astra.algorithm.create(self.cfg_proj)
        astra.algorithm.run(fp_id)
        astra.algorithm.delete(fp_id)
        astra.data2d.delete([vid, sid])
        return s



    def filter_projections(self, proj_set):
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real


    def filter_projections_ext_old(self, sino):
        # Extension with boundaries, can be turned into zero-extension.
        n_angles, n_px = sino.shape
        N = nextpow2(2*n_px)
        isodd = 1 if (n_px & 1) else 0
        sino_extended = np.zeros((n_angles, N))

        sino_extended[:, :n_px] = sino
        boundary_right = (sino[:, -1])[:,np.newaxis]
        sino_extended[:, n_px:(n_px+N)//2] = np.tile(boundary_right, (1, (N-n_px)//2))
        boundary_left = (sino[:, 0])[:,np.newaxis]
        sino_extended[:, (n_px+N)//2:N] = np.tile(boundary_left, (1, (N-n_px)//2+isodd))

        ramp = 1./(N/2) * np.hstack((np.arange(N/2), np.arange(N//2, 0, -1)))
        sino_extended_f = np.fft.fft(sino_extended, N, axis=1)

        return np.fft.ifft(ramp * sino_extended_f, axis=1)[:,:n_px].real


    def filter_projections_ext(self, sino, method=1):
        """
        Filter the projections after extending the sinogram.
        This is often used in local tomography where the sinogram is "cropped",
        to avoid truncation artifacts.

        sino: numpy.ndarray
            sinogram, its shape must be consistend with the current tomography configuration
        method: integer
            if 1, extend the sinogram with its boundaries
            if 0, extend the sinogram with zeros
        """
        n_angles, n_px = sino.shape
        N = nextpow2(2*n_px)
        sino_extended = np.zeros((n_angles, N))
        sino_extended[:, :n_px] = sino
        boundary_right = (sino[:, -1])[:,np.newaxis]
        boundary_left = (sino[:, 0])[:,np.newaxis]
        # TODO : more methods
        method = int(bool(method))
        boundary_left *= method
        boundary_right *= method

        sino_extended[:, n_px:n_px+n_px//2] = np.tile(boundary_right, (1, n_px//2))
        sino_extended[:, -n_px//2:] = np.tile(boundary_left, (1, n_px//2))

        ramp = 1./(N//2) * np.hstack((np.arange(N//2), np.arange(N//2, 0, -1)))
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


    def lambda_tomo(self, sino, mu=0):
        """
        Performs a Lambda-tomography reconstruction, i.e a linear combination
        of the plain BP and the Laplace-BP.
        """

        return self.backproj(convolve1d(sino, np.array([-1, 2, -1])), filt=False) + mu * self.backproj(sino, filt=False)


    def fbp(self, sino, padding=None):
        """
        Runs the Filtered Back-Projection algorithm on the provided sinogram.

        sino : numpy.ndarray
            sinogram. Its shape must be consistent with the current tomography configuration
        padding: integer
            Disabled (None) by default.
            If 0, the sinogram is extended with zeros.
            If 1, the sinogram is extended with its boundaries.
        """
        if padding is None:
            return self.backproj(sino, filt=True)
        else:
            return self.backproj(sino, filt=True, ext=True, method=padding)


    def cleanup(self):
        #~ astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.proj_id)


def clipCircle(x):
    return astra.extrautils.clipCircle(x)



