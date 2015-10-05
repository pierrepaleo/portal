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
import astra

__all__ = ['AstraToolbox', 'clipCircle']



class AstraToolbox:
    '''
        ASTRA toolbox wrapper.
    '''

    def __init__(self, n_pixels, n_angles, rot_center=None, super_sampling=None):
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
        else: angles = n_angles

        self.vol_geom = astra.create_vol_geom(n_x, n_y)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, n_pixels, angles)
        if rot_center:
            self.proj_geom['option'] = {'ExtraDetectorOffset': (rot_center - n_x / 2.) * np.ones(n_angles)}
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)

        # For new projector. Based on wrapper of "sirtfbp"
        # vg : Volume geometry
        self.vg = astra.projector.volume_geometry(self.proj_id)
        # pg : projection geometry
        self.pg = astra.projector.projection_geometry(self.proj_id)

        # ---- Configure Projector ------
        # sinogram shape
        self.sshape = astra.functions.geom_size(self.pg)
        # resulting sinogram (data and identifier)
        self.res_sino = np.zeros(self.sshape, dtype=np.float32)
        self.sid = astra.data2d.link('-sino', self.pg, self.res_sino)
        # Configure projector
        self.cfg_proj = astra.creators.astra_dict('FP_CUDA')
        self.cfg_proj['ProjectionDataId'] = self.sid
        self.cfg_proj['ProjectorId'] = self.proj_id
        if super_sampling:
            self.cfg_proj['option'] = {'DetectorSuperSampling':super_sampling}

        # ---- Configure Backprojector ------
        # volume shape
        self.vshape = astra.functions.geom_size(self.vg)
        # resulting volume (data and identifier)
        self.res_slice = np.zeros(self.vshape, dtype=np.float32)
        self.vid = astra.data2d.link('-vol', self.vg, self.res_slice) # TODO : garb. collection here ?
        # Configure backprojector
        self.cfg_backproj = astra.creators.astra_dict('BP_CUDA')
        self.cfg_backproj['ReconstructionDataId'] = self.vid
        self.cfg_backproj['ProjectorId'] = self.proj_id
        if super_sampling:
            self.cfg_backproj['option'] = {'PixelSuperSampling':super_sampling}

    def backproj_old(self, sino_data, filt=False, ext=False):
        if filt is True:
            if ext is True:
                bid, rec = astra.create_backprojection(self.filter_projections_ext(sino_data), self.proj_id)
            else:
                bid, rec = astra.create_backprojection(self.filter_projections(sino_data), self.proj_id)
        else:
            bid, rec = astra.create_backprojection(sino_data, self.proj_id)
        astra.data2d.delete(bid)
        return rec

    def proj_old(self, slice_data):
        sid, proj_data = astra.create_sino(slice_data, self.proj_id)
        astra.data2d.delete(sid)
        return proj_data


    def backproj(self, s, filt=False, ext=False):
        if filt is True:
            if ext is True:
                sino = self.filter_projections_ext(s).astype(np.float32)
            else:
                sino = self.filter_projections(s).astype(np.float32)
        else:
            sino = s.astype(np.float32)

        sid = astra.data2d.link('-sino', self.pg, sino)
        self.cfg_backproj['ProjectionDataId'] = sid
        bp_id = astra.algorithm.create(self.cfg_backproj)
        astra.algorithm.run(bp_id)
        astra.algorithm.delete(bp_id)
        astra.data2d.delete(sid) # self.vid
        return self.res_slice


    def proj(self, v):
        vid = astra.data2d.link('-vol',self.vg, v)
        self.cfg_proj['VolumeDataId'] = vid
        fp_id = astra.algorithm.create(self.cfg_proj)
        astra.algorithm.run(fp_id)
        astra.algorithm.delete(fp_id)
        astra.data2d.delete(vid) # self.sid
        return self.res_sino


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


def clipCircle(x):
    return astra.extrautils.clipCircle(x)



