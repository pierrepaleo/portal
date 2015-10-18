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
from portal.operators.misc import power_method
from portal.operators.image import norm2sq

__all__ = ['fista_l1']

def fista_l1(data, K, Kadj, Lambda, H, Hinv, soft_thresh, Lip=None, n_it=100, return_all=True):
    '''
    Beck-Teboulle's forward-backward algorithm to minimize the objective function
        ||K*x - d||_2^2 + Lambda*||H*x||_1
    When K and H are linear operators, and H is invertible.

    K : forward operator
    Kadj : backward operator
    Lambda : weight of the regularization (the higher Lambda, the more sparse is the solution in the H domain)
    H : *invertible* linear operator (eg. sparsifying transform, like Wavelet transform).
    Hinv : inverse operator of H
    soft_thresh : *in-place* function doing the soft thresholding (proximal operator of L1 norm) of the coefficients H(image)
    Lip : largest eigenvalue of Kadj*K
    n_it : number of iterations
    return_all: if True, an array containing the values of the objective function will be returned
    '''

    # Check if H, Hinv and soft_thresh are callable
    if not callable(H) or not callable(Hinv) or not callable(soft_thresh): raise ValueError('fista_l1() : the H, Hinv and soft_thresh parameters be callable')
    # Check if H and Hinv are inverse of eachother
    u = np.random.rand(512, 512)
    Hu = H(u)
    if np.max(np.abs(u - Hinv(Hu))) > 1e-3: # FIXME: not sure what tolerance I should take
        raise ValueError('fista_l1() : the H operator inverse does not seem reliable')
    # Check that soft_thresh is an in-place operator
    thr = soft_thresh(Hu, 1.0)
    if thr is not None: raise ValueError('fista_l1(): the soft_thresh parameter must be an in-place modification of the coefficients')
    # Check if the coefficients H(u) have a method "norm1"
    can_compute_l1 = True if callable(getattr(Hu, "norm1", None)) else False

    if Lip is None:
        print("Warn: fista_l1(): Lipschitz constant not provided, computing it with 20 iterations")
        Lip = power_method(K, Kadj, data, 20)**2 * 1.2 # FIXME : why squared ?
        print("Lip = %e" % Lip)

    if return_all: en = np.zeros(n_it)
    x = np.zeros_like(Kadj(data))
    y = np.zeros_like(x)
    for k in range(0, n_it):
        grad_y = Kadj(K(y) - data) # FIXME : Kadj(K(y) - data) !!
        x_old = x
        w = H(y - (1.0/Lip)*grad_y)
        soft_thresh(w, Lambda/Lip)
        x = Hinv(w)
        y = x + ((k-1.0)/(k+10.1))*(x - x_old) # TODO : see what would be the best parameter "a"
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            l1 = w.norm1() if can_compute_l1 else 0.
            energy = fidelity + Lambda*l1
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t L1 %e" % (k, energy, fidelity, l1))
        elif (k%10 == 0): print("Iteration %d" % k)
    if return_all: return en, x
    else: return x
