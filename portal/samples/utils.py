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

import numpy as np
import os.path



#~ def get_data_path():
    #~ """
    #~ setuptools does not install data in site-packages, to avoid namespace pollution.
    #~ This function recovers the portal-data directory, assuming that the prefix is
    #~ of the form .../lib/pythonX.Y/site-packages/... ; the portal-data directory is then
    #~ at the same level as "lib".
    #~ """
    #~ import portal
    #~ d = portal.__path__[0]
    #~ s = os.path.split(d)
    #~ while s[1] != '' and s[1] != 'lib':
        #~ s = os.path.split(d)
        #~ d = s[0]
    #~ if s[1] == '':
        #~ raise RuntimeError("could not find portal-data directory")
    #~ else:
        #~ return os.path.join(s[0], 'portal-data')


def load_lena():
    """
    Loads Lena from a npz file, to avoid a dependency on scipy.misc
    """
    try:
        import scipy.misc
        l = scipy.misc.lena()
    except ImportError:
        #~ datapath = get_data_path()
        #~ fname = os.path.join(datapath, 'lena.npz')
        fname = 'lena.npz'
        f = np.load(fname)
        l = f['data']
        f.close()
    return l.astype(np.float32)



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
