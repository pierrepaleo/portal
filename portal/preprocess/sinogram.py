#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of ESRF nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE




# ------------------------------------------------------------------------------
# ------------Cupping reduction : "sinogram straightening" ---------------------
# ------------------------------------------------------------------------------



def straighten_sino(sino):
    n_angles, n_pix = sino.shape
    x = np.arange(n_pix)
    sino_corr = np.zeros_like(sino)

    i = 0
    for line in range(n_angles):
        y = sino[line, :]
        # Least-Squares, 3rd order polynomial :
        #~ X = np.array([np.ones(n_pix), x, x**2, x**3]).T
        #~ X = X[:, ::-1] # numpy convention
        #~ z0 = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        z = np.polyfit(x, y, 3)
        f = np.poly1d(z)
        sino_corr[line, :] = y - f(x)
    return sino_corr





# ------------------------------------------------------------------------------
# ------------ Determine the center of rotation --------------------------------
# ------------------------------------------------------------------------------

#TODO : implement nhgia's double-wedge technique



# ------------------------------------------------------------------------------
#  Trick to customize the center of rotation in ASTRA.
#  This should not be necessary in future versions of ASTRA.
#  Taken from tomopy :
#
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
# ------------------------------------------------------------------------------


def center_sino(sino0, center):
    sino = np.copy(sino0)
    _, ndet = sino.shape
    shft = int(np.round(ndet / 2. - center))
    sino[:] = np.roll(sino, shft)
    l = shft
    r = sino.shape[1] + shft
    if l < 0:
        l = 0
    if r > sino.shape[1]:
        r = sino.shape[1]
    sino[:, 0:l] = 0
    sino[:, r:sino.shape[1]] = 0
    return sino
















