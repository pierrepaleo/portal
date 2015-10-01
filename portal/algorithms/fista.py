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

from __future__ import division
import numpy as np
from portal.operators.misc import power_method
from portal.operators.image import norm2sq

__all__ = ['fista_l1']

def fista_l1(data, K, Kadj, H, Lambda, Lip=None,  n_it=100, return_energy=True):
    '''
    Beck-Teboulle's forward-backward algorithm to minimize the objective function
        ||K*x - d||_2^2 + ||H*x||_1
    When K and H are linear operators, and H is invertible.

    K : forward operator
    Kadj : backward operator
    H : *invertible* linear operator (eg. sparsifying transform, like Wavelet transform).
        This operator is a class which must have the following methods:
        H.forward(x), or H(x)   : compute the coefficients from the image
        H.soft_threshold()      : in-place soft-thresholding
        H.inverse()             : inversion, from coefficients to image
        H.norm1()               : [optionnal] L1 norm of the coefficients

    Lambda : weight of the regularization (the higher Lambda, the more sparse is the solution in the H domain)
    Lip : norm of the operator K (sqrt of largest eigenvalue of Kadj*K)
    n_it : number of iterations
    return_energy: if True, an array containing the values of the objective function will be returned
    '''


    # Check if the operator H satisfies the requirements
    if not callable(getattr(H, "inverse", None)): raise ValueError('fista_l1() : the H parameter should have a inverse() callable method')
    if not callable(getattr(H, "soft_threshold", None)): raise ValueError('fista_l1() : the H parameter should have a soft_threshold() callable method')
    can_compute_l1 = True if callable(getattr(H, "norm1", None)) else False
    u = np.random.rand(512, 512)
    if np.max(np.abs(u - H(u).inverse())) > 1e-3: # FIXME: not sure what tolerance I should take
        raise ValueError('fista_l1() : the H operator inverse does not seem reliable')

    if Lip is None:
        print("Warn: fista_l1(): Lipschitz constant not provided, computing it with 20 iterations")
        Lip = power_method(K, Kadj, data, 20) * 1.2
        print("Lip = %e" % Lip)

    if return_energy: en = np.zeros(n_it)
    x = np.zeros_like(Kadj(data))
    y = np.zeros_like(x)
    for k in range(0, n_it):
        grad_y = Kadj(K(x) - data)
        x_old = x
        w = H(y - (1.0/Lip)*grad_y)
        w.soft_threshold(Lambda/Lip)
        x = w.inverse()

        #~ import matplotlib.pyplot as plt
        #~ plt.figure()
        #~ plt.imshow(x); plt.colorbar()
        #~ plt.show()

        y = x + ((k-1.0)/(k+10.1))*(x - x_old)
        # Calculate norms
        if return_energy:
            fidelity = 0.5*norm2sq(K(x)-data)
            l1 = w.norm1() if can_compute_l1 else 0.
            energy = fidelity + Lambda*l1
            en[k] = energy
            if (k%10 == 0): # TODO: more flexible
                print("[%d] : energy %e \t fidelity %e \t L1 %e" % (k, energy, fidelity, l1))
    if return_energy: return en, x
    else: return x
