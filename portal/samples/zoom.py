import numpy as np
import portal


def bin2(im):
    return (im[::2, ::2] + im[1::2, ::2] + im[::2, 1::2] + im[1::2, 1::2])/2.

    # im.reshape(l.shape[0]/2,2,l.shape[1]/2,2).sum(axis=1).sum(axis=-1)

def antibin2(im):
    res = np.zeros((2*im.shape[0], 2*im.shape[1]))
    res[::2, ::2] = im
    res[::2, 1::2] = im
    res[1::2, ::2] = im
    res[1::2, 1::2] = im
    return res/2.


if __name__ == '__main__':

    import scipy.misc
    l = scipy.misc.lena()
    lb = l[::2, ::2]

    K = bin2
    Kadj = antibin2

    K = lambda x : bin2(bin2(x))
    Kadj = lambda x : antibin2(antibin2(x))

    '''
    # TV-zooming
    #------------
    Lambda = 0.5
    en, res = portal.algorithms.chambollepock.chambolle_pock_tv(lb, K, Kadj, Lambda, n_it=31, return_energy=True)

    #~ print(res-antibin2(lb)).max()


    '''


    # Wavelets-zooming
    # Laisser converger un peu pour que le random_shifts se debarasse des artefacts
    Lambda =1.1 # quite good with db4, L=0.5
    H = lambda x : portal.operators.wavelets.WaveletCoeffs(x, wname='db4', levels=4, do_random_shifts=True)
    Hinv = lambda w : w.inverse()
    soft_thresh = lambda w, beta : portal.operators.wavelets.soft_threshold_coeffs(w, Lambda)
    en, res = portal.algorithms.fista.fista_l1(lb, K, Kadj, Lambda, H, Hinv, soft_thresh, n_it=91)

    portal.utils.misc.my_imshow([res, Kadj(lb)], shape=(1,2), cmap="gray")


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(en)
    plt.show()
