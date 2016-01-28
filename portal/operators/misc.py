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
from portal.operators.image import norm2sq, dot

__all__ = ['power_method', 'check_adjoint']

def power_method(K, Kadj, data, n_it=10):
    '''
    Calculates the norm of operator K
    i.e the sqrt of the largest eigenvalue of K^T*K
        ||K|| = sqrt(lambda_max(K^T*K))

    K : forward operator
    Kadj : backward operator (adjoint of K)
    data : initial data
    '''
    x = np.copy(Kadj(data)) # Copy in case of Kadj = Id
    for k in range(0, n_it):
        x = Kadj(K(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)



def check_adjoint(K, Kadj, K_input_shape, Kadj_input_shape):
    '''
    Checks if the operators K and Kadj are actually adjoint of eachother, i.e if
        < K(x), y > = < x, Kadj(y) >
    '''

    if len(K_input_shape) == 1:
        x = np.random.rand(K_input_shape[0])
    elif len(K_input_shape) == 2:
        x = np.random.rand(K_input_shape[0], K_input_shape[1])
    elif len(K_input_shape) == 3:
        x = np.random.rand(K_input_shape[0], K_input_shape[1], K_input_shape[2])

    if len(Kadj_input_shape) == 1:
        y = np.random.rand(Kadj_input_shape[0])
    elif len(Kadj_input_shape) == 2:
        y = np.random.rand(Kadj_input_shape[0], Kadj_input_shape[1])
    elif len(Kadj_input_shape) == 3:
        y = np.random.rand(Kadj_input_shape[0], Kadj_input_shape[1], Kadj_input_shape[2])

    err = abs(dot(K(x), y) - dot(x, Kadj(y)))
    return err


