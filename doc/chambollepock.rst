Chambolle-Pock algorithm in PORTAL : a turotial
###############################################


.. include:: texenv.txt


The Chambolle-Pock algorithm in PORTAL
=========================================

PORTAL implements a Chambolle-Pock solver for Total Variation regularization.
It can solve problems of the type

.. raw:: html

    $$
    \amin{x}{\frac{1}{2} \norm{A x - b}_2^2 + \lambda \norm{\nabla x}_1}
    $$

where :math:`A` is specified by the user. The advantage of using the Chambolle-Pock algorithm for this kind of problem is that each step is made of simple element-wise operations.
This would not have been true for a FISTA Total Variation solver with general operator :math:`A:`.

Example : Total Variation denoising
------------------------------------

Here, the operator :math:`A` is simply the identity. The syntax of ``chambolle_pock_tv`` is the following


.. code-block:: python

    import portal

    # Create the noisy image (30% of the max value)
    import scipy.misc
    l = scipy.misc.lena().astype('f')
    pc = 0.3
    lb = l + np.random.rand(l.shape[0], l.shape[1])* l.max() * pc

    # Define the operator and its adjoint
    Id = lambda x : x
    K = Id
    Kadj = Id
    Lambda = 20.

    res  = portal.algorithms.chambollepock.chambolle_pock_tv(lb, A, Aadj, Lambda, n_it=101, return_all=False)
    portal.utils.misc.my_imshow([lb, res], shape=(1,2), cmap="gray")

If the norm :math:`L` of :math:`K = \begin{bmatrix} A ,  \nabla \end{bmatrix}` is not provided, ``chambolle_pock_tv`` automatically computes it.


.. container:: twocol

    .. container:: leftside

        .. figure:: images/lena_gaussian_noise.png
            :scale: 50
            :align: center
            :alt: Lena with gaussian noise
            :figclass: align-center

            Lena with gaussian noise, 30% of maximum value

    .. container:: rightside

        .. figure:: images/lena_gaussian_noise_cp.png
            :scale: 50
            :align: center
            :alt: Lena denoised
            :figclass: align-center

            Lena denoised with Chambolle-Pock TV solver


The ``chambollepock.chambolle_pock_l1_tv`` function can also be used for L1-TV minimization.
This is useful for noise containing strong outliers (eg. salt & pepper noise)

.. container:: twocol

    .. container:: leftside

        .. figure:: images/lena_sp_noise.png
            :scale: 50
            :align: center
            :alt: Lena with salt and pepper noise
            :figclass: align-center

            Lena with salt and pepper noise

    .. container:: rightside

        .. figure:: images/lena_sp_noise_cp.png
            :scale: 50
            :align: center
            :alt: Lena denoised
            :figclass: align-center

            Lena denoised with Chambolle-Pock TV solver




Example : Total Variation deblurring
-------------------------------------

Here, the operator :math:`A` is a blurring operator. It can be implemented with a convolution with a Gaussian kernel.
PORTAL implements the convolution operator (with any 1D or 2D kernel) and its adjoint.

.. code-block:: python

    import portal

    sigma = 2.6

    # Define the operator A and its adjoint
    gaussian_kernel = portal.utils.misc.gaussian1D(sigma) # Faster separable convolution
    Blur = portal.operators.convolution.ConvolutionOperator(gaussian_kernel)
    A = lambda x : Blur*x
    Aadj = lambda x : Blur.adjoint() * x

    # Create the blurred image
    import scipy.misc
    l = scipy.misc.lena().astype('f')
    lb = A(l)

    Lambda = 5e-2
    res  = portal.algorithms.chambollepock.chambolle_pock_tv(lb, A, Aadj, Lambda, n_it=501, return_all=False)
    portal.utils.misc.my_imshow([lb, res], shape=(1,2), cmap="gray")

(note that here it takes more iterations to converge, and the regularization parameter is much smaller than in the denoising case).

PORTAL can also help to determine if ``A`` and ``Aadj`` are actually adjoint of eachother -- an important property for the algorithm.

.. code-block:: python

    portal.operators.misc.check_adjoint(A, Aadj, lb.shape, lb.shape)


.. container:: twocol

    .. container:: leftside

        .. figure:: images/lena_gaussian_blur.png
            :scale: 50
            :align: center
            :alt: Lena with gaussian blur
            :figclass: align-center

            Lena with gaussian blur

    .. container:: rightside

        .. figure:: images/lena_gaussian_blur_cp.png
            :scale: 50
            :align: center
            :alt: Lena deblurred
            :figclass: align-center

            Lena deblurred with Chambolle-Pock TV solver




Example : Total Variation tomographic reconstruction
-----------------------------------------------------

Here, the operator :math:`A` is the forward tomography projector. PORTAL uses the ASTRA toolbox to compute the forward and backward projector.
For performances issues (the forward and backward projectors are implemented on GPU), the operators :math:`A` and :math:`A^T` are not exactly matched (i.e adjoint of eachother).
In practice, this is not an issue for the reconstruction.

.. code-block:: python

    import portal

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
    en, res = portal.algorithms.chambollepock.chambolle_pock_tv(sino, K, Kadj, Lambda, L=22.5, n_it=301, return_all=True)

    # Display the result, compare to FBP
    res_fbp = Kadj(sino)
    portal.utils.misc.my_imshow((res_fbp, res), (1,2), cmap="gray", nocbar=True)


.. container:: twocol

    .. container:: leftside

        .. figure:: images/lena_tomo_fbp.png
            :scale: 50
            :align: center
            :alt: Lena reconstructed with 80 projections, Filtered Backprojection
            :figclass: align-center

            Lena reconstructed with 80 projections, Filtered Backprojection

    .. container:: rightside

        .. figure:: images/lena_tomo_tv.png
            :scale: 50
            :align: center
            :alt: Lena reconstructed with 80 projections, TV regularization
            :figclass: align-center

            Lena reconstructed from 80 projections with TV minimization





*Note* : the ASTRA toolbox comes with many available geometries ; but in PORTAL only the parallel geometry has been wrapped.





Mathematical background
========================

Presentation of the algorithm
-----------------------------


The Chambolle-Pock algorithm is a very versatile method to solve various optimization problems.

Suppose you want to solve the problem

.. raw:: html

    $$
    \amin{x}{F(x) + G(K x)}
    $$

Or, equivalently

.. raw:: html

    $$
    \underset{x}{\min} \mmax{y}{ \braket{K x}{y} + F(x) - G^* (y) }
    $$

where :math:`F` and :math:`G` are convex (possibly non smooth) and :math:`K` is a linear operator.

The general form of the basic Chambolle-Pock algorithm can be written :

.. raw:: html

    $$
    \begin{aligned}
    y_{n+1} &= \prox{\sigma G^*}{y_n + \sigma K \tilde{x}_n} \\
    x_{n+1} &= \prox{\tau F}{x_n - \tau K^* y_{n+1}} \\
    \tilde{x}_{n+1} &= x_{n+1} + \theta (x_{n+1} - x_n)
    \end{aligned}
    $$

The primal step size :math:`\tau` and the dual step size :math:`\sigma` should be chosen such that
:math:`\sigma \tau \leq 1/L^2` where :math:`L` is the norm of the operator :math:`K`.


This algorithm is not a proximal gradient descent -- no gradient is computed here.
This is a *primal-dual* method, performing one step in the primal domain (prox of :math:`F`) and one step in the dual domain (prox of :math:`G^*`) ;
a kind of combination of Douglas-Rachford (fully primal) and ADMM (fully dual).

Chambolle-Pock algorithm is actually much more versatile than proximal gradient algorithms --
an even more flexible algorithm is described
`here
<http://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/publis/Condat-optim-SPL-2014.pdf>`_.
All you need is defining an operator :math:`K`, the functions :math:`F`, :math:`G` and their proximal.
Computing the proximal of :math:`F` or :math:`G` is not straightforward in general.
When this cannot be done in one step, there are two solutions : re-write the optimization problem (see next section) or split again :math:`F` and :math:`G` like in the aforementioned algorithm.


Deriving the algorithm for L2-TV
------------------------------------

The Total Variation regularized image deblurring can be written

.. raw:: html

    $$
    \amin{x}{\frac{1}{2}\norm{A x - b}_2^2 + \lambda \norm{\nabla x}_1}
    $$

where :math:`A` is a linear operator.
An attempt to solve this problem with the Chambolle-Pock algorithm would be to write

.. raw:: html

    $$
    F(x) = \frac{1}{2} \norm{A x - b}_2^2 \qquad \text{and} \qquad G(x) = \lambda \norm{x}_1 \qquad \text{and} \qquad K = \nabla
    $$

However, the proximal operator of :math:`F` is

.. raw:: html

    $$
    \prox{\tau F}{y} = \left( \id + \tau A^T A \right)^{-1} \left( y + \tau A^T b \right)
    $$

so it involves the computation of the inverse of :math:`\id + \tau A^T A`, which is an ill-posed problem.
This inverse can be computed if :math:`A` is a convolution (since it is diagonalized by the Fourier Transform), for example in the deblurring case,
but this is not the case in general.

We'll consider the case in which :math:`A^T A` is not easily invertible. The optimization problem has to be rewritten. We'll make use of the following equalities :

.. raw:: html

    $$
    \frac{1}{2} \norm{A x - b}_2^2 = \mmax{q}{\braket{A x - b}{q} - \frac{1}{2} \norm{q}_2^2}
    $$

and

.. raw:: html

    $$
    \begin{aligned}
    \lambda \norm{\nabla x}_1 &= \mmax{\norm{z}_\infty \leq 1}{\braket{\nabla x}{\lambda z}} \\
    &= \mmax{\norm{z}_\infty \leq \lambda}{\braket{\nabla x}{z}} \\
    &= \mmax{z}{\braket{\nabla x}{z} - i_{B_\infty^\lambda} (z)}
    \end{aligned}
    $$

so the initial problem can be rewritten :

.. raw:: html

    $$
    \underset{x}{\min} \mmax{q, z}
    {
    \braket{A x - b}{q}
    + \braket{\nabla x}{z}
    - \frac{1}{2} \norm{q}_2^2
    - i_{B_\infty^\lambda} (z)
    }
    $$

Noting that

.. raw:: html

    $$
    \begin{aligned}
    \braket{A x}{q} + \braket{\nabla x}{z} &= \braket{x}{A^T q} + \braket{x}{\nabla^* z} \\
    &= \braket{x}{A^T q + \nabla^* z} \\
    &= \braket{x}{\begin{bmatrix} A^T & \nabla^* \end{bmatrix} \pmat{q \\ z} } \\
    &= \braket{K x}{\pmat{q \\ z}}
    \end{aligned}
    $$

with :math:`K = \begin{bmatrix} A \\ \nabla \end{bmatrix}`

the problem becomes

.. raw:: html

    $$
    \underset{x}{\min} \mmax{q, z}
    {
    \braket{K x}{\pmat{q \\ z}}
    + F(x)
    - G^*(q, z)
    }
    $$
    where we can identify
    $$
    \begin{aligned}
    F(x) &= 0  \\
    G^* (q, z) &= \braket{b}{q} + \frac{1}{2} \norm{q}_2^2 + i_{B_\infty^\lambda} (z) \\
    K &= \begin{bmatrix} A \\ \nabla \end{bmatrix}
    \end{aligned}
    $$

which is the saddle-point formulation of the problem.
Here, the proximal of :math:`F` is the identity, and the proximal of :math:`G^*` is separable with respect to :math:`q` and :math:`z` (since :math:`G^*` is a separable sum of these variables).
Its computation is straightforward :

.. raw:: html

    $$
    \prox{\sigma G^*}{q, z} = \pmat{\frac{q - \sigma b}{1 + \sigma} \; , \; \proj{B_\infty^\lambda}{z}}
    $$

where :math:`\proj{B_\infty^\lambda}{z}` is the projection onto the L-:math:`\infty` ball of radius :math:`\lambda`, which is an elementise operation.

Eventually, the Chambolle-Pock algorithm for this problem is :

.. raw:: html

    $$
    \begin{aligned}
    q_{n+1} &= \left( q_n + \sigma A \tilde{x}_n - \sigma b \right) / (1 + \sigma) \\
    z_{n+1} &= \proj{B_\infty^\lambda}{ z_n + \sigma \nabla \tilde{x}_n } \\
    x_{n+1} &= x_n - \tau A^T q_{n+1} + \tau \div{z_{n+1}} \\
    \tilde{x}_{n+1} &= x_{n+1} + \theta (x_{n+1} - x_n)
    \end{aligned}
    $$

Prototyping algorithms in the primal-dual framework is more difficult than for proximal gradient algorithms ; but it enables much more flexibility.
With PORTAL, the user just has to specify the linear operator for a fixed regularization.
