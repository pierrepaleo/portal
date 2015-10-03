import numpy as np
import portal


if __name__ == '__main__':

    fname ='/home/paleo/Projets/ID16/A10_noOs_tomo2/sino/sino_1501.edf'
    folder_out = '/home/paleo/Projets/ID16/A10_noOs_tomo2/rec_sirt'

    sino = portal.utils.io.edf_read(fname)
    n_angles, n_px = sino.shape
    AST = portal.operators.tomography.AstraToolbox(n_px, n_angles, rot_center=1049.371362)

    FBP = lambda x : AST.backproj(x, filt=True)
    P = lambda x : AST.proj(x)

    sino = portal.preprocess.sinogram.straighten_sino(sino)
    #~ sino = portal.preprocess.sinogram.center_sino(sino, 1049.4)

    res = FBP(sino)

    S = portal.algorithms.sirtfilter.SirtFilter(n_px, n_angles, 57, savedir=folder_out, lambda_tikhonov=0.0, rot_center=1049.371362)
    res_sirt = S.reconst(sino)
    #~ res_sirt = AST.run_algorithm('SIRT_CUDA', 100, sino)

    #~ import matplotlib.pyplot as plt
    #~ plt.figure()
    #~ plt.imshow(FBP(sino))
    #~ plt.show()

    #~ portal.utils.io.call_imagej(res)
    portal.utils.io.call_imagej([res, res_sirt])
