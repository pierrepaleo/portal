#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD 2-clause Simplified
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
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


from setuptools import setup, find_packages
import os
#~ import sys
#~ import glob


if __name__ == '__main__':


    cmdclass = {}

    packages = ['portal', 'portal.test', 'portal.utils', 'portal.preprocess', 'portal.operators', 'portal.algorithms', 'portal.samples']
    package_dir = {"portal": "portal",
            "portal.test": "test",
            'portal.utils': 'portal/utils',
            'portal.preprocess':'portal/preprocess',
            'portal.operators':'portal/operators',
            'portal.algorithms': 'portal/algorithms',
            'portal.samples': 'portal/samples'}

    setup(name = "portal",
        version = "0.1",
        platforms = ["linux_x86", "linux_x86_64"],
        description = "A set of various tomographic reconstruction algorithms",
        author = "Pierre Paleo",
        author_email = "pierre.paleo@esrf.fr",
        maintainer = "Pierre Paleo",
        maintainer_email = "pierre.paleo@esrf.fr",
        url = "https://github.com/pierrepaleo/portal",
        license="BSD",
        #~ packages=find_packages(),
        #~ packages=find_packages(exclude=("test", ),

        packages=packages,
        package_dir = package_dir,
        package_data={'': ['samples/lena.npz']},
        #~ data_files=[("portal-data", ["portal/samples/lena.npz"])], # not copied in site-packages !

        long_description = """
        This module contains iterative algorithms for tomographic reconstruction. These algorithms
        minimize a cost function containing a L2 (least-squares) fidelity term, and a sparsity-promoting prior.
        The module also contains various pre-processing routines like rings artifacts removal.
        """
        )


