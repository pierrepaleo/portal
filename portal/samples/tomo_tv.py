#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import portal


def _main():

    # Create phantom
    import scipy.misc
    l = scipy.misc.lena().astype(np.float32)
    ph = portal.utils.misc.phantom_mask(l)

    # Create Projector and Backprojector
    npx = l.shape[0]
    nangles = 80
    AST = portal.operators.tomography.AstraToolbox(npx, nangles)

    # Configure the TV regularization
    Lambda = 5.0

    # Configure the optimization algorithm (Chambolle-Pock for TV min)
    K = lambda x : AST.proj(x)
    Kadj = lambda x : AST.backproj(x, filt=True)
    n_it = 101

    # Run the algorithm to reconstruct the sinogram
    sino = K(ph)
    en, res = portal.algorithms.chambollepock.chambolle_pock_tv(sino, K, Kadj, Lambda, L=22.5, n_it=301, return_all=True)

    # Display the result, compare to FBP
    res_fbp = Kadj(sino)
    portal.utils.misc.my_imshow((res_fbp, res), (1,2), cmap="gray", nocbar=True)
    scipy.misc.imsave("lena_tomo_fbp.png", res_fbp)
    scipy.misc.imsave("lena_tomo_tv.png", res)


if __name__ == '__main__':
    _main()
