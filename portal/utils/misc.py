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
try:
    import matplotlib.pyplot as plt
    __has_plt__ = True
except ImportError:
    __has_plt__ = False
from math import ceil

__all__ = ['gaussian1D', 'generate_coords', 'my_imshow', 'phantom_mask', 'phantom']

if __has_plt__:
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
else:
    def my_imshow(img_list, shape=None, cmap=None, nocbar=False):
        raise ImportError("Please install matplotlib to use this function.")



#~ def generate_coords(img,center=None):
    #~ if center is None: center = img.shape[0]/2, img.shape[1]/2
    #~ R, C = np.mgrid[0:img.shape[0],0:img.shape[1]]
    #~ R -= center[0]; C -= center[1]
    #~ return R, C

def generate_coords(img_shp, center=None):
    """
    Credits : E. Gouillart
    """
    l_r, l_c = float(img_shp[0]), float(img_shp[1])
    R, C = np.mgrid[:l_r, :l_c]
    if center is None:
        center0, center1 = l_r / 2., l_c / 2.
    else:
        center0, center1 = center
    R += 0.5 - center0
    C += 0.5 - center1
    return R, C




def from_gradient_to_image(gr, f00=0):
    '''
    Reconstruct an image from its gradient.

    gr : gradient of the image : (2, N, N) numpy array
    f00 : value of f(0, 0)
    '''

    # Reconstruct the first column from f(0, 0)
    col0 = np.cumsum((gr[0])[:,0])
    col0[1:] = col0[:-1]
    col0 = col0 + f00

    # Reconstruct the image from the first column
    res = np.empty_like(gr[0])
    res[:, 0] = col0
    res[:, 1:] = gr[1][:, :-1]
    return np.cumsum(res, axis=1)


def from_gradient_comp_to_image(gr, boundary=None, axis=1, renorm=False):
    '''
    Reconstruct an image from only one component of its gradient.
    Since only one component is available, a whole boundary has to be provided,
    either the first line (for comonent 0)  or column (for component 1) of the image.

    gr : one component of the gradient : (N, N) numpy ndarray
    boundary : prior knowledge on the image : first line or first column.
    axis : component (0: lines, 1: columns)
    '''

    if len(gr.shape) > 2:
        raise ValueError('It seems that more of one gradient component were provided. If so, please use from_gradient_to_image()')

    if boundary is None:
        print("Warning : boundary condition not provided. Solution might be incoherent along axis %d" % (1-axis))
        res = np.cumsum(gr, axis=axis)
        #
        if renorm is True:
            mn = np.mean(res, axis=1)
            res2 = np.copy(res)
            for i in range(res.shape[0]):
                res2[i, :] = res[i, :] - mn[i]
            res = res2
        #
        return res

    else:
        if len(boundary.shape) > 1:
            raise ValueError('Please provide a one-dimensional boundary (first line or first column)')
        if boundary.shape[0] != gr.shape[axis]:
            raise ValueError('The boundary length does not match the length of dimension %d' % axis)

        # TODO : extend for axis 0 (transpose)
        res = np.empty_like(gr)
        res[:, 0] = boundary
        res[:, 1:] = gr[:, :-1]
        return np.cumsum(res, axis=1)


def _generate_center_coordinates(l_x):
    """
    Compute the coordinates of pixels centers for an image of
    linear size l_x
    """
    l_x = float(l_x)
    X, Y = np.mgrid[:l_x, :l_x]
    center = l_x / 2.
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y



def fit_fourier(s_comp, c_comp):

    Nr, Nc = s_comp.shape
    S = np.fft.fftshift(np.fft.fft2(s_comp))
    C = np.fft.fftshift(np.fft.fft2(c_comp))
    #~ kr, kc = portal.utils.misc.generate_coords(s_comp)
    kr, kc = _generate_center_coordinates(Nr)
    renorm = 1.0/(kr**2 + kc**2)

    renorm[np.isinf(renorm)] = 1.0 # /!\

    F_real = kr * S.imag + kc * C.imag
    F_real *= renorm
    F_imag = -kr * S.real - kc * C.real
    F_imag *= renorm

    return np.fft.ifft2(np.fft.ifftshift((F_real + 1j*F_imag))).real





def ceilpow2(N):
    p = 1
    while p < N:
        p *= 2
    return p


def fftbs(x):
    '''
    Bluestein chirp-Z transform.
    Computes a FFT on a "good" length (power of two) to speed-up the FFT, without extending the spectrum.
    '''
    N = x.shape[0]
    n = np.arange(N)
    b = np.exp((1j*pi*n**2)/N)
    a = x * b.conjugate()
    M = ceilpow2(N) * 2
    #~ A = np.concatenate((a, [0] * (M - N)))
    B = np.concatenate((b, [0] * (M - 2*N + 1), b[:0:-1]))
    C = np.fft.ifft(np.fft.fft(a, M) * np.fft.fft(B))

    c = C[:N]
    return b.conjugate() * c















def phantom_mask(img, radius=None):
    if radius is None: radius = img.shape[0]//2-10
    R, C = generate_coords(img.shape)
    M = R**2+C**2
    res = np.zeros_like(img)
    res[M<radius**2] = img[M<radius**2]
    return res

def clip_circle(img, center=None, radius=None):
    R, C = generate_coords(img.shape, center)
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

