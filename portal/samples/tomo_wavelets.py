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

    # Configure the wavelet regularization
    wname = "haar"
    levels = 8
    do_random_shifts = False
    lambda_ = 1.0

    # Configure the optimization algorithm
    K = lambda x : AST.proj(x)
    Kadj = lambda x : AST.backproj(x, filt=True)
    H = portal.operators.wavelets.WaveletCoeffs


    sino = K(ph)
    res_fbp = Kadj(sino)

    en, res = portal.algorithms.fista.fista_l1(sino, K, Kadj, H, Lambda=2., Lip=None,  n_it=351, return_energy=True)

    portal.utils.misc.my_imshow((res_fbp, res), (1,2), cmap="gray", nocbar=True)


if __name__ == '__main__':
    _main()
