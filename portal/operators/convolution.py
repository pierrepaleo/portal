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
from scipy.ndimage import filters

__all__ = ['ConvolutionOperator']

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






