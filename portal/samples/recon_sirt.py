#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import portal

"""
In this example, we test the "SIRT-FBP" approach for tomographic reconstruction.
SIRT is a L2 iterative method. When applying gradient descent, it turns out that
the slice at iteration n is a linear combination of the slice at iteration n-1.
Thus, the slice at iteration n can be directly expressed from the data,
enabling a "filter backprojection"-like algorithm replacing SIRT.

See Pelt, D. M. and Batenburg, K. J., "Accurately approximating algebraic tomographic
reconstruction by filtered backprojection", Proceedings of the 2015 International
Meeting on Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine
"""

if __name__ == '__main__':

    # Generate a 256x256 phantom, project it with 32 angles
    npx, na = 256, 32
    tomo = portal.operators.tomography.AstraToolbox(npx, na)
    pht = portal.utils.misc.phantom(npx).astype(np.float32)
    pht_p = tomo.proj(pht)

    # Build the SIRT-FBP filter with 200 iterations
    S = portal.algorithms.sirtfilter.SirtFilter(npx, na, 200)#, savedir='./')

    # Apply this filter to the projected phantom, compare to FBP
    portal.utils.misc.my_imshow([tomo.backproj(pht_p, filt=True), S.reconst(pht_p)], shape=(1,2), cmap="gray")

