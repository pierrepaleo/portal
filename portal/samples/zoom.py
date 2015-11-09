#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import portal.operators.wavelets
import portal.algorithms
from portal.samples.utils import load_lena
import portal.utils


"""
In this example, we use the Wavelet regularization for digital zooming.

Many operators can be used for the zoom, from the simplest 2x2-binning to filtering (like Lanczos filtering).
Here we take a 2x2 binning, which can be extended to 4x4, 8x8, ... by successive applications of this procedure.

The 2x2 binning and its adjoint are straightforward to define.
Regarding the wavelet regularization, the parameter depends on how many time we want to zoom.
A good choice of wavelets id a wavelet with a few vanishing moments, like Daubechies 4.

"""






def bin2(im):
    return (im[::2, ::2] + im[1::2, ::2] + im[::2, 1::2] + im[1::2, 1::2])/2.

def antibin2(im):
    res = np.zeros((2*im.shape[0], 2*im.shape[1]))
    res[::2, ::2] = im
    res[::2, 1::2] = im
    res[1::2, ::2] = im
    res[1::2, 1::2] = im
    return res/2.


if __name__ == '__main__':

    l = load_lena()

    # 2-zoom
    #~ K = bin2
    #~ Kadj = antibin2

    # 4-zoom
    K = lambda x : bin2(bin2(x))
    Kadj = lambda x : antibin2(antibin2(x))

    lb = K(l) # Use lb = l  instead of lb = K(l)  for "true" digital zooming of lena (slower !)

    '''
    # TV-zooming
    #------------
    Lambda = 0.5
    en, res = portal.algorithms.chambollepock.chambolle_pock_tv(lb, K, Kadj, Lambda, n_it=31, return_energy=True)

    #~ print(res-antibin2(lb)).max()


    '''


    # Wavelets-zooming
    # Laisser converger un peu pour que le random_shifts se debarasse des artefacts
    #Lambda =1.5 # quite good with db4, L=0.5, 2-bin, levels=4
    Lambda = 15.0
    H = lambda x : portal.operators.wavelets.WaveletCoeffs(x, wname='db4', levels=4, do_random_shifts=True)
    Hinv = lambda w : w.inverse()
    soft_thresh = lambda w, beta : portal.operators.wavelets.soft_threshold_coeffs(w, Lambda)
    en, res = portal.algorithms.fista.fista_l1(lb, K, Kadj, Lambda, H, Hinv, soft_thresh, n_it=41)

    portal.utils.misc.my_imshow([res, Kadj(lb)], shape=(1,2), cmap="gray")


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(en)
    plt.show()
