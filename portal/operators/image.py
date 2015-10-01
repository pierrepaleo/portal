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

__all__ = ['gradient', 'div', 'gradient_axis', 'div_axis', 'tv_smoothed', 'grad_tv_smoothed', 'proj_l2', 'norm2sq', 'norm1', 'dot', 'entropy', 'KL']


def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


def div(grad):
    '''
    Compute the divergence of a gradient
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    '''
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


def gradient_axis(x, axis=-1):
    '''
    Compute the gradient (keeping dimensions) along one dimension only.
    By default, the axis is -1 (diff along columns).
    '''
    t1 = np.empty_like(x)
    t2 = np.empty_like(x)
    if axis != 0:
        t1[:, :-1] = x[:, 1:]
        t1[:, -1] = 0
        t2[:, :-1] = x[:, :-1]
        t2[:, -1] = 0
    else:
        t1[:-1, :] = x[1:, :]
        t1[-1, :] = 0
        t2[:-1, :] = x[:-1, :]
        t2[-1, :] = 0
    return t1-t2



def div_axis(x, axis=-1):
    '''
    Compute the opposite of divergence (keeping dimensions), adjoint of the gradient, along one dimension only.
    By default, the axis is -1 (div along columns).
    '''
    t1 = np.empty_like(x)
    t2 = np.empty_like(x)
    if axis != 0:
        t1[:, :-1] = -x[:, :-1]
        t1[:, -1] = 0
        t2[:, 0] = 0
        t2[:, 1:] = x[:, :-1]
    else:
        t1[:-1,: ] = -x[:-1, :]
        t1[-1, :] = 0
        t2[0, :] = 0
        t2[1:, :] = x[:-1, :]
    return t1 + t2

def psi(x, mu):
    '''
    Huber function needed to compute tv_smoothed
    '''
    res = np.abs(x)
    m = res < mu
    #~ if (m.sum() > 1): print(m.sum()) # Debug
    res[m] = x[m]**2/(2*mu) + mu/2
    return res


def tv_smoothed(x, mu):
    '''
    Moreau-Yosida approximation of Total Variation
    see Weiss, Blanc-Féraud, Aubert, "Efficient schemes for total variation minimization under constraints in image processing"
    '''
    g = gradient(x)
    g = np.sqrt(g[0]**2 + g[1]**2)
    return np.sum(psi(g, mu))

def grad_tv_smoothed(x, mu):
    '''
    Gradient of Moreau-Yosida approximation of Total Variation
    '''
    g = gradient(x)
    g_mag = np.sqrt(g[0]**2 + g[1]**2)
    m = g_mag >= mu
    m2 = (m == False) #bool(1-m)
    #~ if (m2.sum() > 30): print(m2.sum()) # Debug
    g[0][m] /= g_mag[m]
    g[1][m] /= g_mag[m]
    g[0][m2] /= mu
    g[1][m2] /= mu
    return -div(g)


def proj_l2(g, Lambda=1.0):
    '''
    Proximal operator of the L2,1 norm :
        L2,1(u) = sum_i ||u_i||_2
    i.e pointwise projection onto the L2 unit ball

    g : gradient-like numpy array
    Lambda : magnitude of the unit ball
    '''
    res = np.copy(g)
    n = np.maximum(np.sqrt(np.sum(g**2, 0))/Lambda, 1.0)
    res[0] /= n
    res[1] /= n
    return res



def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())

def norm1(mat):
    return np.sum(np.abs(mat))

def dot(mat1, mat2):
    return np.dot(mat1.ravel(), mat2.ravel())




def entropy(img):
    '''
    Computes the entropy of an image (similar to Matlab function)
    '''
    h, _ = np.histogram(img, 256)
    h = h.astype('f')
    h /= 1.0*img.size
    h[h == 0] = 1.0
    return -np.sum(h*np.log(h))

def KL(img1, img2):
    '''
    Computes the Kullback-Leibler divergence between two images
    Mind that this function is not symmetric. The second argument should be the "reference" image.
    '''
    x, _ = np.histogram(img1, 256)
    y, _ = np.histogram(img2, 256)
    m = (y != 0) # integers
    x_n, y_n = x[m], y[m]
    m = (x_n != 0)
    x_n, y_n = 1.0 * x_n[m], 1.0 * y_n[m]
    Sx, Sy = x.sum()*1.0, y.sum()*1.0
    return (1.0/Sx) * np.sum(x_n * np.log(x_n/y_n * Sy/Sx))
