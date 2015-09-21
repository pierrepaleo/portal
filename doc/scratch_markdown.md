$$
\newcommand{\braket}[2]{\left\langle #1 \, , \, #2 \right\rangle}
\renewcommand{\div}{\operatorname{div}}
\newcommand{\norm}[1]{\left\| #1 \right\|}
\newcommand{\mmax}[2]{\underset{#1}{\max}\left\{ #2 \right\}}
\newcommand{\mmin}[2]{\min_{#1}\left\{ #2 \right\}}
$$

Approximate
$$
J(x) = \norm{\nabla x}_1 = \mmax{\norm{z}_{\infty}\leq 1}{\braket{\nabla x}{z}}
$$
by
$$
J_\mu (x) = \mmax{\norm{z}_{\infty}\leq 1}{\braket{\nabla x}{z} - \frac{\mu}{2}\norm{z}_2^2} + \frac{n\mu}{2} %huber-like function
$$
Then $J_{\mu} (x) = \displaystyle\sum_i \psi_{\mu}\left( |(\nabla x)_i| \right)$
$$
\psi_{\mu}(x) =
\begin{cases}
|v| & \text{if } |v| \geq \mu \\
\frac{v^2}{2\mu} + \frac{\mu}{2} & \text{otherwise}
\end{cases}
$$
and $\nabla J_{\mu} (x) = -\div{\Phi}\quad $ with
$$
\Phi_i =
\begin{cases}
\frac{(\nabla x)_i}{|(\nabla x)_i|} & \text{if } |(\nabla x)_i| \geq \mu \\
\frac{(\nabla x)_i}{\mu} & \text{otherwise}
\end{cases}
$$


