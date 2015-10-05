import numpy as np
import portal

if __name__ == '__main__':

    npx, na = 256, 32
    AST = portal.operators.tomography.AstraToolbox(npx, na)
    pht = portal.utils.misc.phantom(npx).astype(np.float32)

    pht_p = AST.proj(pht)

    S = portal.algorithms.sirtfilter.SirtFilter(npx, na, 800)#, savedir='./')

    portal.utils.misc.my_imshow(S.reconst(pht_p), cmap="gray")

