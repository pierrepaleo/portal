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



import os
import sys

# Current version -- Major.Minor.Patch (see http://semver.org)
__version__ = '0.1.1'




#~
#~ def _isImportedFromSourceDirectory():
    #~ """Returns True if this module is imported from its source directory."""
#~
    #~ # sys.path[0]: script directory or '' for interactive interpreter
    #~ path = os.path.abspath(sys.path[0])
    #~ # filePath == __file__ if __file__ is absolute (See os.path.join).
    #~ filePath = os.path.join(path, __file__)
    #~ return os.path.commonprefix((path, filePath)) == path
#~
#~
#~ if _isImportedFromSourceDirectory():
    #~ raise ImportError('Cannot be imported from source directory.')


######################

# Astra for tomographic projection-backprojection
try:
    import astra
    __has_astra__ = True
except ImportError:
    print("Warning: ASTRA was not found. Tomographic reconstruction algorithms will not work. ASTRA can be found at https://github.com/astra-toolbox/astra-toolbox")
    __has_astra__ = False

# pywt for Wavelet utils
try:
    import pywt
    __has_pywt__ = True
except ImportError:
    print("Warning: pywt was not found. Wavelets-related algorithms will not work. pywt can be simply installed using pip install pywt  or  sudo apt-get install python-pywt")
    __has_pywt__ = False

# scipy.ndimage for convolutions in direct space
try:
    import scipy.ndimage
    __has_ndimage__ = True
except ImportError:
    print("Warning: scipy.ndimage was not found. Convolutions with small kernels may be slow.")
    __has_ndimage__ = False

# h5py for exporting sirt-filter
try:
    import h5py
    __has_h5py__ = True
except ImportError:
    print("Warning: h5py was not found. SIRT-filters will only be exported into .npz format")
    __has_h5py__ = False


######################


from portal import algorithms
from portal import operators
from portal import preprocess
from portal import utils




#~ del _isImportedFromSourceDirectory  # Clean-up module namespace
