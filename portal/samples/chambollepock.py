#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import portal.algorithms
import portal.operators.convolution
import portal.utils
from portal.samples.utils import load_lena, sp_noise


"""
In this file, the Chambolle-Pock algorithm is used for a few problems :
   - TV-L2 denoising
   - TV-L2 deblurring
   - TV-L1 denoising
   - TV-L1 tomographic reconstruction
"""



def _main(CASE, DO_RETURN_ALL):

    l = load_lena()
    SAVE = False


    if CASE == 1: # Denoising (Gaussian noise)

        pc = 0.3
        lb = l + np.random.rand(l.shape[0], l.shape[1])*l.max() * pc
        Lambda = 17.

        # With custom C-P
        #~ res = chambolle_pock(lb, Lambda, 101)
        Id = lambda x : x
        K = Id
        Kadj = Id
        res  = portal.algorithms.chambollepock.chambolle_pock_tv(lb, K, Kadj, Lambda, L= 3.5, n_it=151, return_all=DO_RETURN_ALL)
        if DO_RETURN_ALL: res = res[1]
        portal.utils.misc.my_imshow([lb, res], shape=(1,2), cmap="gray")
        if SAVE:
            scipy.misc.imsave("lena_gaussian_noise.png", lb)
            scipy.misc.imsave("lena_gaussian_noise_cp.png", res)

    if CASE == 2: # deblurring (Gaussian blur)

        sigma = 2.6
        # Define the operator A and its adjoint
        gaussian_kernel = portal.utils.misc.gaussian1D(sigma)
        Blur = portal.operators.convolution.ConvolutionOperator(gaussian_kernel)
        A = lambda x : Blur*x
        Aadj = lambda x : Blur.adjoint() * x

        # Create the blurred image
        lb = A(l)

        Lambda = 0.03
        res  = portal.algorithms.chambollepock.chambolle_pock_tv(lb, A, Aadj, Lambda, n_it=801, return_all=DO_RETURN_ALL)
        if DO_RETURN_ALL: res = res[1]
        portal.utils.misc.my_imshow([lb, res], shape=(1,2), cmap="gray")
        if SAVE:
            scipy.misc.imsave("lena_gaussian_blur.png", lb)
            scipy.misc.imsave("lena_gaussian_blur_cp.png", res)


    if CASE == 3: # denoising (salt & pepper noise)

        lb = sp_noise(l, 0.1, 0.1)
        Lambda = 0.7
        Id = lambda x : x
        A = Id
        Aadj = Id

        res  = portal.algorithms.chambollepock.chambolle_pock_l1_tv(lb, A, Aadj, Lambda, n_it=1001, return_all=DO_RETURN_ALL)
        if DO_RETURN_ALL: res = res[1]
        portal.utils.misc.my_imshow([lb, res], shape=(1,2), cmap="gray")
        if SAVE:
            scipy.misc.imsave("lena_sp_noise.png", lb)
            scipy.misc.imsave("lena_sp_noise_cp.png", res)
        # Here many iterations are required.
        # Warning : isotropic TV is zero at the beginning ; this is an entirely non-smooth problem !

    if CASE == 4: # tomographic reconstruction with TV-L1
        n_proj = 80
        tomo = portal.operators.tomography.AstraToolbox(l.shape[0], n_proj)
        A = lambda x : tomo.proj(x)
        Aadj = lambda x : tomo.backproj(x, filt=True)
        portal.operators.tomography.clipCircle(l)
        sino = A(l) * 1.5708/n_proj

        Lambda = 0.1
        res  = portal.algorithms.chambollepock.chambolle_pock_tv(sino, A, Aadj, Lambda, L=2.5e1, n_it=301, return_all=DO_RETURN_ALL)
        if DO_RETURN_ALL: res = res[1]
        fbp = Aadj(sino)
        #
        portal.operators.tomography.clipCircle(fbp)
        portal.operators.tomography.clipCircle(res)
        #
        portal.utils.misc.my_imshow([fbp, res], shape=(1,2), cmap="gray")





if __name__ == '__main__':

    CASE = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    if len(sys.argv) >= 3:
        DO_RETURN_ALL = bool(int(sys.argv[2]))
    else: DO_RETURN_ALL = False

    _main(CASE, DO_RETURN_ALL)



