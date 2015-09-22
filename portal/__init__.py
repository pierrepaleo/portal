#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD 2-clause Simplified
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
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import os
import sys


def _isImportedFromSourceDirectory():
    """Returns True if this module is imported from its source directory."""

    # sys.path[0]: script directory or '' for interactive interpreter
    path = os.path.abspath(sys.path[0])
    # filePath == __file__ if __file__ is absolute (See os.path.join).
    filePath = os.path.join(path, __file__)
    return os.path.commonprefix((path, filePath)) == path


if _isImportedFromSourceDirectory():
    raise ImportError('Cannot be imported from source directory.')

'''
from portal.algorithms.chambollepock import *
from portal.algorithms.conjgrad import *
from portal.algorithms.sirtfilter import *

from portal.operators.convolution import *
from portal.operators.tomography import *
from portal.operators.image import *
from portal.operators.misc import *

from portal.preprocess.rings import *
from portal.preprocess.sinogram import *

from portal.utils.io import *
from portal.utils.misc import *
'''


from portal import algorithms
from portal import operators
from portal import preprocess
from portal import utils

#~ from portal.algorithms.chambollepock import chambolle_pock
#~ from portal.algorithms.conjgrad import conjugate_gradient_TV
#~ from portal.algorithms.sirtfilter import SirtFilter

#~ from portal.operators.convolution import ConvolutionOperator
#~ from portal.operators.tomography import AstraToolbox
#~ from portal.operators import image
#~ from portal.operators import misc

#~ from portal.preprocess import rings
#~ from portal.preprocess import sinogram

#~ from portal.utils import io
#~ from portal.utils import misc




del _isImportedFromSourceDirectory  # Clean-up module namespace
