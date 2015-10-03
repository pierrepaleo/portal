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
    en, res = portal.algorithms.chambollepock.chambolle_pock_tv(sino, K, Kadj, Lambda, n_it=351, return_energy=True)

    # Display the result, compare to FBP
    res_fbp = Kadj(sino)
    portal.utils.misc.my_imshow((res_fbp, res), (1,2), cmap="gray", nocbar=True)


if __name__ == '__main__':
    _main()
