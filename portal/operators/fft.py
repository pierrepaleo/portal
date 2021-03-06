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
from multiprocessing import cpu_count
#~ # pycuda
#~ try:
    #~ import pycuda
    #~ __has_pycuda__ = True
#~ except ImportError:
    #~ __has_pycuda__ = False
    #~ __has_skcuda__ = False
#~ # skcuda
#~ if __has_pycuda__:
    #~ try:
        #~ import skcuda
        #~ __has_skcuda__ = True
    #~ except ImportError:
        #~ __has_skcuda__ = False
# pyfftw
try:
    import pyfftw
    __has_pyfftw__ = True
except ImportError:
    __has_pyfftw__ = False
#
#~ if __has_skcuda__:
    #~ import pycuda.autoinit
    #~ import pycuda.gpuarray as gpuarray
    import skcuda
    #~ import skcuda.fft as cufft



class Fft():
    """
    Create a FFT plan corresponding to a given input array.

    Parameters
    ----------
    inarray : numpy.ndarray
        Input array from which the plan is created
    outarray : numpy.ndarray
        Output array
    threads : int
        If using FFTW, number of threads which should be used for the computation of FFT
    axis : tuple of ints
    """

    def __init__(self, inarray, outarray=None, threads=None, axis=None, force_complex=False):
        self.plan_f = None
        self.plan_i = None
        self.np_f = None # Forward transform when using numpy
        self.np_i = None # Inverse transform when using numpy
        self.r2c = True
        self.nthreads = -1
        self.__get_types(inarray, outarray)
        if force_complex: self.r2c = False

        ndims = inarray.ndim
        if ndims > 3: ndims = 3 # for looking in the dictionary below

        if __has_pyfftw__:
            # Number of threads to be used
            ncpus = cpu_count()
            self.nthreads = ncpus # by default with FFTW, use the maximum number of threads. The GIL is released !
            if threads: # user-defined number of threads
                if threads > ncpus: print("Warning: fft(): Requested more threads (%d) than available CPUs (%d)" % (threads, ncpus))
                elif threads < 0: threads = ncpus
                else: self.nthreads = threads

            # Determine the shape of the output array
            if outarray: out_shape = outarray.shape
            else:
                if self.r2c: # (N1, ..., Np/2+1)
                    out_shape = list(inarray.shape)
                    out_shape[-1] = out_shape[-1]//2+1
                    out_shape = tuple(out_shape)
                else: out_shape = inarray.shape
            self.inarray = np.copy(inarray) # input can be destroyed !
            if force_complex: self.inarray = self.inarray + 0j
            self.outarray = np.zeros(out_shape, dtype=self.otype)

            # Create the plans
            # pyfftw.builders has issues for reall FFT on odd sizes
            #~ transforms = {
                #~ (1, True): (pyfftw.builders.rfft, pyfftw.builders.irfft),
                #~ (2, True): (pyfftw.builders.rfft2, pyfftw.builders.irfft2),
                #~ (3, True): (pyfftw.builders.rfftn, pyfftw.builders.irfftn),
                #~ (1, False): (pyfftw.builders.fft, pyfftw.builders.ifft),
                #~ (2, False): (pyfftw.builders.fft2, pyfftw.builders.ifft2),
                #~ (3, False): (pyfftw.builders.fftn, pyfftw.builders.ifftn)
            #~ }
            #~ plan_builder = transforms[(ndims, self.r2c)]
            #~ self.plan_f = plan_builder[0](self.inarray, planner_effort='FFTW_ESTIMATE', threads=self.nthreads) # TODO: handle "axes" property for ndims > 1
            #~ self.plan_i = plan_builder[1](self.outarray, planner_effort='FFTW_ESTIMATE', threads=self.nthreads)

            # ATTENTION : pyfftw does not perform ND fft by default (see axes default value)
            nd_axes = tuple(np.arange(self.inarray.ndim)) # TODO : handle user-defined "axes" property
            self.plan_f = pyfftw.FFTW(self.inarray, self.outarray, axes=nd_axes, direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), threads=self.nthreads)
            self.plan_i = pyfftw.FFTW(self.outarray, self.inarray, axes=nd_axes, direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', ), threads=self.nthreads)

        else: # Only numpy is available
            transforms = {
                (1, True): (np.fft.rfft, np.fft.irfft),
                (2, True): (np.fft.rfft2, np.fft.rfft2),
                (3, True): (np.fft.rfftn, np.fft.rfftn),
                (1, False): (np.fft.fft,  np.fft.fft),
                (2, False): (np.fft.fft2, np.fft.fft2),
                (3, False): (np.fft.fftn, np.fft.fftn)
            }
            self.np_f = transforms[(ndims, self.r2c)][0]
            self.np_i = transforms[(ndims, self.r2c)][1]


    def __get_types(self, inarray, outarray=None):

        types_in_out = {np.float32: np.complex64, np.float64: np.complex128, np.complex64: np.complex64, np.complex128: np.complex128}
        types_in_out = {np.dtype(k): np.dtype(v) for k, v in types_in_out.iteritems()}
        types_out_in = {v: k for k, v in types_in_out.iteritems()}

        if inarray.dtype not in types_in_out.keys():
            raise ValueError("fft(): Unsupported input format %s" % str(inarray.dtype))
        self.itype = inarray.dtype
        self.r2c = self.isreal(self.itype)

        if outarray is not None:
            if outarray.dtype not in types_out_in.keys():
                raise ValueError("fft(): Unsupported output format %s" % str(outarray.dtype))
            if (types_out_in[outarray.dtype] != self.itype):
                raise ValueError("fft(): Output format for %s should be %s (got %s)" %(str(self.itype), str(types_in_out[self.itype]), str(outarray.dtype)))
            self.otype = outarray.dtype
        else:
            self.otype = types_in_out[self.itype]


    def fft(self, arr):
        if __has_pyfftw__:
            if arr.shape != self.inarray.shape:
                raise ValueError("fft(): provided array has shape %s when plan was created for shape %s" % (str(arr.shape), str(self.inarray.shape)))
            if arr.dtype != self.itype:
                raise ValueError("fft(): provided array has type %s when plan was created for type %s" % (str(arr.dtype), str(self.inarray.dtype)))
            #~ if (id(arr) != id(self.inarray)):
            new_input = np.copy(arr)
            if self.isreal(arr.dtype) and not(self.r2c): # "force_complex"
                new_input = new_input + 0j
            self.plan_f.update_arrays(new_input, self.outarray)
            self.plan_f.execute()
            return self.plan_f.get_output_array()
        else:
            return self.np_f(arr)



    def ifft(self, arr):
        if __has_pyfftw__:
            if arr.shape != self.outarray.shape:
                raise ValueError("ifft(): provided array has shape %s when plan was created for shape %s" % (str(arr.shape), str(self.outarray.shape)))
            if arr.dtype != self.otype:
                raise ValueError("ifft(): provided array has type %s when plan was created for type %s" % (str(arr.dtype), str(self.outarray.dtype)))
            new_output = np.copy(arr) # !!
            self.plan_i.update_arrays(new_output, self.inarray)
            self.plan_i.execute()
            res = self.plan_i.get_output_array()
            # unlike pyfftw.builders/numpy.fft, the result has to be normalized !
            return res/self.plan_i.N
        else:
            return self.np_i(arr)


    @staticmethod
    def isreal(thetype):
        return thetype in [np.float32, np.float64]



