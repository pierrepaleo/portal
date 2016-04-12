#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from portal.operators.tomography import AstraToolbox
from portal.algorithms.fista import fista_l1_operator
from portal.samples.utils import load_lena
import portal.utils
# -----


USE_PDWT = 1

if USE_PDWT:
    from pypwt import Wavelets
else:
    from portal.operators.wavelets import WaveletCoeffs as Wavelets, soft_threshold_coeffs


def _main():

    # Create phantom
    l = load_lena()
    ph = portal.utils.misc.phantom_mask(l)

    # Add noise
    ph += np.random.randn(*ph.shape)*0.1*ph.max()

    # Create Projector and Backprojector
    npx = l.shape[0]
    nangles = 80
    AST = AstraToolbox(npx, nangles)

    # Configure the wavelet regularization
    wname = "db2"
    levels = 8
    do_random_shifts = True
    Lambda = 90 if USE_PDWT else 0.2
    Lambda = 130.0 # if noise

    # Configure the optimization algorithm (FISTA-L2-L1)
    K = lambda x : AST.proj(x)
    Kadj = lambda x : AST.backproj(x, filt=True)
    if not(USE_PDWT):
        H = lambda x : Wavelets(x, wname=wname, levels=levels, do_random_shifts=do_random_shifts)
        Hinv = lambda w : w.inverse()
        soft_thresh = lambda w, beta : soft_threshold_coeffs(w, Lambda)
    else:
        W = Wavelets(l, wname, levels, do_swt=1)
        #~ W.info()
        def H(x):
            W.forward(x)
            return W.coeffs
        def Hinv(w_useless):
            W.inverse()
            return W.image
        def soft_thresh(w_useless, beta):
            W.soft_threshold(beta, 0)

    n_it = 201

    # Run the algorithm to reconstruct the sinogram
    sino = K(ph)
    en, res = fista_l1_operator(sino, K, Kadj,
        Lambda=Lambda, H=H, Hinv=Hinv, soft_thresh=soft_thresh,
        Lip=None, n_it=n_it, return_all=True)

    # Display the result, compare to FBP
    res_fbp = Kadj(sino)
    portal.utils.misc.my_imshow((res_fbp, res), (1,2), cmap="gray", nocbar=True)


if __name__ == '__main__':
    _main()
