#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import portal.operators.wavelets
import portal.algorithms
from portal.samples.utils import load_lena
import portal.utils

def _main():

    # Create phantom
    l = load_lena()
    ph = portal.utils.misc.phantom_mask(l)

    # Create Projector and Backprojector
    npx = l.shape[0]
    nangles = 80
    AST = portal.operators.tomography.AstraToolbox(npx, nangles)

    # Configure the wavelet regularization
    wname = "db4"
    levels = 8
    do_random_shifts = True
    Lambda = 0.2

    # Configure the optimization algorithm (FISTA-L2-L1)
    K = lambda x : AST.proj(x)
    Kadj = lambda x : AST.backproj(x, filt=True)
    H = lambda x : portal.operators.wavelets.WaveletCoeffs(x, wname=wname, levels=levels, do_random_shifts=do_random_shifts)
    Hinv = lambda w : w.inverse()
    soft_thresh = lambda w, beta : portal.operators.wavelets.soft_threshold_coeffs(w, Lambda)
    n_it = 201

    # Run the algorithm to reconstruct the sinogram
    sino = K(ph)
    en, res = portal.algorithms.fista.fista_l1_operator(sino, K, Kadj,
        Lambda=Lambda, H=H, Hinv=Hinv, soft_thresh=soft_thresh,
        Lip=None, n_it=n_it, return_all=True)

    # Display the result, compare to FBP
    res_fbp = Kadj(sino)
    portal.utils.misc.my_imshow((res_fbp, res), (1,2), cmap="gray", nocbar=True)


if __name__ == '__main__':
    _main()
