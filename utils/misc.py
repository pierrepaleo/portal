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

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def my_imshow(img_list, shape=None, cmap=None, nocbar=False):
    if isinstance(img_list, np.ndarray):
        is_array = True
        plt.figure()
        plt.imshow(img_list, interpolation="nearest", cmap=cmap)
        if nocbar is False: plt.colorbar()
        plt.show()

    elif shape:
        num = np.prod(shape)
        #~ if num > 1 and is_array:
            #~ print('Warning (my_imshow): requestred to show %d images but only one image was provided' %(num))
        if num != len(img_list):
            raise Exception('ERROR (my_imshow): requestred to show %d images but %d images were actually provided' %(num , len(img_list)))

        plt.figure()
        for i in range(0, num):
            curr = str(shape + (i+1,))
            curr = curr[1:-1].replace(',','').replace(' ','')
            if i == 0: ax0 = plt.subplot(curr)
            else: plt.subplot(curr, sharex=ax0, sharey=ax0)
            plt.imshow(img_list[i], interpolation="nearest", cmap=cmap)
            if nocbar is False: plt.colorbar()
        plt.show()




def generate_coords(img,center=None):
    if center is None: center = img.shape[0]/2, img.shape[1]/2
    R, C = np.mgrid[0:img.shape[0],0:img.shape[1]]
    R -= center[0]; C -= center[1]
    return R, C

def phantom_mask(img, radius=None):
    if radius is None: radius = img.shape[0]//2-10
    R, C = generate_coords(img)
    M = R**2+C**2
    res = np.zeros_like(img)
    res[M<radius**2] = img[M<radius**2]
    return res



def gaussian1D(sigma):
    ksize = int(ceil(8 * sigma + 1))
    if (ksize % 2 == 0): ksize += 1
    t = np.arange(ksize) - (ksize - 1.0) / 2.0
    g = np.exp(-(t / sigma) ** 2 / 2.0).astype('f')
    g /= g.sum(dtype='f')
    return g


################################################################################
# Phantoms utils. by Alex Opie  <lx_op@orcon.net.nz>
################################################################################


## Copyright (C) 2010  Alex Opie  <lx_op@orcon.net.nz>
##
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.


def phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
        """
         phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)

        Create a Shepp-Logan or modified Shepp-Logan phantom.

        A phantom is a known object (either real or purely mathematical)
        that is used for testing image reconstruction algorithms.  The
        Shepp-Logan phantom is a popular mathematical model of a cranial
        slice, made up of a set of ellipses.  This allows rigorous
        testing of computed tomography (CT) algorithms as it can be
        analytically transformed with the radon transform (see the
        function `radon').

        Inputs
        ------
        n : The edge length of the square image to be produced.

        p_type : The type of phantom to produce. Either
          "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
          if `ellipses' is also specified.

        ellipses : Custom set of ellipses to use.  These should be in
          the form
                [[I, a, b, x0, y0, phi],
                 [I, a, b, x0, y0, phi],
                 ...]
          where each row defines an ellipse.
          I : Additive intensity of the ellipse.
          a : Length of the major axis.
          b : Length of the minor axis.
          x0 : Horizontal offset of the centre of the ellipse.
          y0 : Vertical offset of the centre of the ellipse.
          phi : Counterclockwise rotation of the ellipse in degrees,
                measured as the angle between the horizontal axis and
                the ellipse major axis.
          The image bounding box in the algorithm is [-1, -1], [1, 1],
          so the values of a, b, x0, y0 should all be specified with
          respect to this box.

        Output
        ------
        P : A phantom image.

        Usage example
        -------------
          import matplotlib.pyplot as pl
          P = phantom ()
          pl.imshow (P)

        References
        ----------
        Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
        from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
        Feb. 1974, p. 232.

        Toft, P.; "The Radon Transform - Theory and Implementation",
        Ph.D. thesis, Department of Mathematical Modelling, Technical
        University of Denmark, June 1996.

        """

        if (ellipses is None):
                ellipses = _select_phantom (p_type)
        elif (np.size (ellipses, 1) != 6):
                raise AssertionError ("Wrong number of columns in user phantom")

        # Blank image
        p = np.zeros ((n, n))

        # Create the pixel grid
        ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]

        for ellip in ellipses:
                I   = ellip [0]
                a2  = ellip [1]**2
                b2  = ellip [2]**2
                x0  = ellip [3]
                y0  = ellip [4]
                phi = ellip [5] * np.pi / 180  # Rotation angle in radians

                # Create the offset x and y values for the grid
                x = xgrid - x0
                y = ygrid - y0

                cos_p = np.cos (phi)
                sin_p = np.sin (phi)

                # Find the pixels within the ellipse
                locs = (((x * cos_p + y * sin_p)**2) / a2
              + ((y * cos_p - x * sin_p)**2) / b2) <= 1

                # Add the ellipse intensity to those pixels
                p [locs] += I

        return p


def _select_phantom (name):
        if (name.lower () == 'shepp-logan'):
                e = _shepp_logan ()
        elif (name.lower () == 'modified shepp-logan'):
                e = _mod_shepp_logan ()
        else:
                raise ValueError ("Unknown phantom type: %s" % name)

        return e


def _shepp_logan ():
        #  Standard head phantom, taken from Shepp & Logan
        return [[   2,   .69,   .92,    0,      0,   0],
                [-.98, .6624, .8740,    0, -.0184,   0],
                [-.02, .1100, .3100,  .22,      0, -18],
                [-.02, .1600, .4100, -.22,      0,  18],
                [ .01, .2100, .2500,    0,    .35,   0],
                [ .01, .0460, .0460,    0,     .1,   0],
                [ .02, .0460, .0460,    0,    -.1,   0],
                [ .01, .0460, .0230, -.08,  -.605,   0],
                [ .01, .0230, .0230,    0,  -.606,   0],
                [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan ():
        #  Modified version of Shepp & Logan's head phantom,
        #  adjusted to improve contrast.  Taken from Toft.
        return [[   1,   .69,   .92,    0,      0,   0],
                [-.80, .6624, .8740,    0, -.0184,   0],
                [-.20, .1100, .3100,  .22,      0, -18],
                [-.20, .1600, .4100, -.22,      0,  18],
                [ .10, .2100, .2500,    0,    .35,   0],
                [ .10, .0460, .0460,    0,     .1,   0],
                [ .10, .0460, .0460,    0,    -.1,   0],
                [ .10, .0460, .0230, -.08,  -.605,   0],
                [ .10, .0230, .0230,    0,  -.606,   0],
                [ .10, .0230, .0460,  .06,  -.605,   0]]

