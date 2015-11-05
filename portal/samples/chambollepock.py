#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import portal

"""
In this file, the Chambolle-Pock algorithm is used for a few problems :
   - TV-L2 denoising
   - TV-L2 deblurring
   - TV-L1 denoising
   - TV-L1 tomographic reconstruction

Modify the variable "CASE" to apply each of these cases.
"""



def sp_noise(img, salt=None, pepper=None):
    '''
    Salt & Pepper noise.
    @param img : image
    @param salt : "salt probability"
    @param pepper : "pepper probability"
    '''
    if salt is None: salt = img.shape[0]
    if pepper is None: pepper = img.shape[0]
    if salt > 1 or pepper > 1 or salt < 0 or pepper < 0:
        raise ValueError("Invalid arguments : salt & pepper must be between 0 and 1")
    salt_bound = np.int(1.0/salt)
    pepper_bound = np.int(1.0/pepper)
    salt_mask = np.random.randint(salt_bound+1, size=img.shape)
    salt_mask = (salt_mask == salt_bound)
    pepper_mask = np.random.randint(pepper_bound+1, size=img.shape)
    pepper_mask = (pepper_mask == pepper_bound)
    res = np.copy(img)
    res[salt_mask] = res.max()
    res[pepper_mask] = res.min()
    return res



if __name__ == '__main__':

    import scipy.misc
    l = scipy.misc.lena().astype('f')

    CASE = 3
    SAVE = True


    if CASE == 1: # Denoising (Gaussian noise)

        pc = 0.3
        lb = l + np.random.rand(l.shape[0], l.shape[1])*l.max() * pc
        Lambda = 17.

        # With custom C-P
        #~ res = chambolle_pock(lb, Lambda, 101)
        Id = lambda x : x
        K = Id
        Kadj = Id
        # Test
        #~ K2 = lambda x : (x, grad(x))
        #~ Kadj2 = lambda u : u[0]- div(u[1])
        _, res  = portal.algorithms.chambollepock.chambolle_pock_tv(lb, K, Kadj, Lambda, L= 3.5, n_it=151)
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
        l = scipy.misc.lena().astype('f')
        lb = A(l)

        Lambda = 0.03
        _, res  = portal.algorithms.chambollepock.chambolle_pock_tv(lb, A, Aadj, Lambda, n_it=801, return_all=True)
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

        _, res  = portal.algorithms.chambollepock.chambolle_pock_l1_tv(lb, A, Aadj, Lambda, n_it=2001, return_all=True)
        portal.utils.misc.my_imshow([lb, res], shape=(1,2), cmap="gray")
        if SAVE:
            scipy.misc.imsave("lena_sp_noise.png", lb)
            scipy.misc.imsave("lena_sp_noise_cp.png", res)
        # Here many iterations are required.
        # Warning : isotropic TV is zero at the beginning ; this is an entirely non-smooth problem !

    if CASE == 4: # tomographic reconstruction with TV-L1
        sino = portal.utils.io.edf_read('enter/the/path/to/sino.edf')
        sino = portal.preprocess.sinogram.straighten_sino(sino)
        tomo = portal.operators.tomography.AstraToolbox(sino.shape[1], sino.shape[0])
        A = lambda x : tomo.proj(x)
        Aadj = lambda x : tomo.backproj(x, filt=True)

        Lambda = 10.0#0.7
        #~ _, res  = portal.algorithms.chambollepock.chambolle_pock_l1_tv(sino, A, Aadj, Lambda, L=5.2e1,  n_it=91, return_all=True)
        _, res  = portal.algorithms.chambollepock.chambolle_pock_tv(sino, A, Aadj, Lambda, L=5.2e1,  n_it=101, return_all=True)
        fbp = Aadj(sino)
        #
        portal.operators.tomography.clipCircle(fbp)
        portal.operators.tomography.clipCircle(res)
        #
        portal.utils.io.call_imagej([fbp, res])









