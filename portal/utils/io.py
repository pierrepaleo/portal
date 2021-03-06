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




try:
    import scipy.io
    __has_scipyio__ = True
except ImportError:
    __has_scipyio__ = False
try:
    import h5py
    __has_hdf5__ = True
except ImportError:
    __has_hdf5__ = False
try:
    from PyMca.EdfFile import EdfFile
    __has_edf__ = True
except ImportError:
    try:
        from PyMca5.PyMca.EdfFile import EdfFile
        __has_edf__ = True
    except ImportError:
        __has_edf__ = False


__all__ = ['edf_read', 'edf_write', 'h5_read', 'h5_write', 'loadmat', 'call_imagej']

if __has_edf__:

    def edf_read(fname):
            '''
            Read a EDF file and store it into a numpy array
            '''
            e = EdfFile(fname)
            return e.GetData(0)

    def edf_write(data, fname, info={}):
            '''
            Save a numpy array into a EDF file
            '''
            edfw = EdfFile(fname, access='w+') # Overwrite !
            edfw.WriteImage(info, data)

else:
    edf_read = None
    edf_write = None

if __has_hdf5__:
    def h5_read(fname):
        fid = h5py.File(fname)
        res = fid[fid.keys()[0]].value
        fid.close()
        return res

    def h5_write(arr, fname):
        raise NotImplementedError('H5 write is not implemented yet')

else:
    h5_read = None
    h5_write = None


def loadmat(fname):
    try:
        res = scipy.io.loadmat(fname)
    except NotImplementedError: # Matlab >= 7.3 files
        res = h5_read(fname)
    return res



