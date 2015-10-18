Installation
============


Installation from the Python wheel
-----------------------------------

A Python wheel is provided for an easy installation. Simply download the wheel (.whl file) and install it with pip :

.. code-block:: bash

    pip install --user wheel.whl

where `wheel.whl` is the wheel of the current version.

If you are *updating* PORTAL, you have to force the re-installation :

.. code-block:: bash

    pip install --user --force-reinstall wheel.whl



Installation from the sources
------------------------------

Alternatively, you can build and install this package from the sources.

.. code-block:: bash

    git clone git://github.com/pierrepaleo/portal

To generate a wheel, go in PORTAL root folder :

.. code-block:: bash

    python setup.py bdist_wheel

The generated wheel can be installed with the aforementioned instructions.


Dependencies
-------------

To use PORTAL, you must have Python > 2.7 and numpy >= 1.8. These should come with standard Linux distributions.

Numpy is the only component absolutely required for PORTAL. For special applications, the following are required :

   * The `ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox/>`_ for tomography applications

   * ``pywt`` for Wavelets applications. This is a python module which can be installed with ``apt-get install python-pywt``

   * ``scipy.ndimage`` is used for convolutions with small kernel sizes. If not installed, all the convolutions are done in the Fourier domain, which can be slow.


**Note** : Python 3.* has not been tested yet.
