# Factored singularities and high-order Lax-Friedrichs sweeping schemes for point-source traveltimes and amplitudes 


#### Abstract

In the high frequency regime, the geometrical-optics approximation for the Helmholtz equation with a point source results in an eikonal equation for traveltime and a transport equation for amplitude. Because the point-source traveltime field has an upwind singularity at the source point, all formally high-order finite-difference eikonal solvers exhibit first-order convergence and relatively large errors. In this paper, we propose to first factor out the singularities of traveltime, takeoff angles, and amplitudes, and then we design high-order Lax-Friedrichs sweeping schemes for pointsource traveltimes, takeoff angles, and amplitudes. Numerical examples are presented to demonstrate the performance of our new method.


## 1 Introduction

In the high frequency regime, the geometrical-optics approximation for the Helmholtz equation with a point source results in an eikonal equation for traveltime and a transport equation for amplitude. Because the point-source traveltime field has an upwind singularity at the source point, all formally highorder finite-difference eikonal solvers exhibit first-order convergence and relatively large errors. Moreover, the resultant inaccuracy in traveltime prevents reliable computations of takeoff angles and amplitudes. In this paper, we propose to first factor out the singularities of traveltime, takeoff angles, and amplitudes; based on this factorization, we design high-order Lax-Friedrichs sweeping schemes for point-source traveltimes, takeoff angles, and amplitudes.

Many finite-difference methods have been introduced to solve the eikonal equation with a point-source condition. In the vast literature, we cite just a few of them to illustrate the point: $[20,19,12,13,16,6,11,22]$. The traveltime field is mostly smooth, suggesting that high-order finite-difference methods should be effective. The use of upwind schemes in all of the cited methods confine the errors to singularities which develop away from the source point. However, the source point itself is a upwind singularity. Thus most of the published high-order eikonal solvers for point-source conditions have to initialize the traveltime field analytically near the source by imposing a grid-independent region of constant

velocity near the source; see [15, 21, 14]. This approach has two essential drawbacks: (1) the velocity may not be homogeneous near the source, and (2) the size of the region of analytic computations must be set by the user and bears no direct relation to the grid parameters. In principle, highly accurate ray-tracing methods may be used to alleviate the first difficulty, but the second remains: it introduces an arbitrary parameter into the use of eikonal solvers. Although the fixed local grid refinement method proposed in [6] compensates for the loss of accuracy near the source point, that method still has an adhoc parameter to be chosen by the user. The adaptive grid refinement method proposed in [11] overcomes these drawbacks successfully, but it incurs a heavy burden in numerical implementation.

To overcome the above difficulties efficiently without any adhoc parameters, based on the observation that near the source the singularities in traveltime, takeoff angle, and amplitude in inhomogeneous media can be well-captured by those singularities in homogeneous media and inspired by the first-order fast sweeping method for the factored eikonal equation as in [3], we propose to factor out the singularities explicitly in either multiplicative or additive manner; based on the resulting factorization we design high-order Lax-Friedrichs sweeping schemes for solving factored eikonal and transport equations. With highorder accurate traveltimes and amplitudes at our disposal, we construct asymptotic wavefields and make comparison with direct solutions of the Helmholtz equation.

The outline of the paper is as follows. In Section 3, we present the factorization on the traveltime, take-off angle and out-of-plane curvature. In Section 4, we present a third-order WENO based Lax-Friedrichs scheme to solve the factored equations. Numerical examples are presented in Section 5 with comparisons to the results obtained by adaptive methods [11]. And we use our results to construct Green functions compared with the results by a Helmholtz solver [2]. Conclusion remarks are given at the end.

# 2 Fundamental equations 

For a source $\left(x_{0}, z_{0}\right)$ in an isotropic solid, the least traveltime $\tau(x, z)$ is the viscosity solution of an eikonal equation $[8,1]$,

$$
|\nabla \tau|=s(x, z)
$$

with the initial condition

$$
\lim _{(x, z) \rightarrow\left(x_{0}, z_{0}\right)}\left(\frac{\tau(x, z)}{\sqrt{\left(x-x_{0}\right)^{2}+\left(z-z_{0}\right)^{2}}}-\frac{1}{v(x, z)}\right)=0
$$

where $v=1 / s$ is the velocity.
Based on the traveltime field, one can approximate the amplitude field by

solving a transport equation [18],

$$
\nabla \tau \cdot \nabla A+\frac{1}{2} A \nabla^{2} \tau=0
$$

Equation (3) is a first-order advection equation for the amplitude $A$. In order to get a first-order accurate amplitude field, one needs a third-order accurate traveltime field since the Laplacian of the traveltime field is involved [11].

Denoting $\phi$ as the take-off angle of a ray from the source point $\left(x_{0}, z_{0}\right)$ to a general point $(x, z)$, it is constant along any ray

$$
\nabla \tau \cdot \nabla \phi=\frac{\partial \tau}{\partial x} \frac{\partial \phi}{\partial x}+\frac{\partial \tau}{\partial z} \frac{\partial \phi}{\partial z}=0
$$

Since the wavefront normal $\nabla \tau$ is tangential to the ray, the gradient $\nabla \phi$ is tangential to the wavefront.

In 2-D isotropic media with line sources, the amplitude satisfies the formula $([18,4])$

$$
A=\frac{v \sqrt{|\nabla \tau \times \nabla \phi|}}{2 \sqrt{2 \pi}}
$$

For a typical seismic point source, one needs to compensate for the out-of-plane radiation in the 2-D line source amplitude formula (5). The 2-D amplitude with a point source (2.5-D amplitude) can be computed by

$$
A=\frac{\sqrt{v \tau_{y y}|\nabla \tau \times \nabla \phi|}}{4 \pi}
$$

where the out-of-plane curvature $\tau_{y y}$ satisfies an advection equation [17],

$$
\frac{\partial \tau}{\partial x} \frac{\partial \tau_{y y}}{\partial x}+\frac{\partial \tau}{\partial z} \frac{\partial \tau_{y y}}{\partial z}+\tau_{y y}^{2}=0
$$

If a first-order accurate amplitude field is required, then the gradients $\nabla \tau$ and $\nabla \phi$ involved in the amplitude formulas should be at least first-order accurate. According to equation (4), at least second-order accurate derivatives of the traveltime $\tau$ are required to get first-order accurate $\nabla \phi$. Therefore, at least third-order accurate traveltime $\tau$ is required to get a first-order amplitude field.

The point-source traveltime $\tau(x, z)$ has an upwind singularity at the source $\left(x_{0}, z_{0}\right)$. Any first-order or high-order finite-difference eikonal solver can formally have first-order convergence and relatively large errors, because the low accuracy near the source can spread out to the whole space. Therefore, to obtain high accuracy in computing point-source traveltimes one has to treat this upwind singularity carefully. One possible approach is the adaptive method proposed in [11], in which the mesh near the source is refined adaptively according to a user-specified threshold in accuracy; as a result, the accuracy loss due to the singularity of the traveltime field near the point source is compensated by adaptive mesh refinement near the source. Another approach to treat this singularity is to explicitly factor out the singularity in traveltime field due

to the point source as first proposed in [3], in which the traveltime is factorized into two multiplicative factors, one of which being able to capture the source singularity explicitly. This factorization results in an underlying function that is smooth in a neighborhood of the source and satisfies a factored eikonal equation; consequently, a first-order fast sweeping scheme yields a fully first-order accurate traveltime field as demonstrated in [3].

In this work, we utilize this factorization idea for the point-source traveltime $\tau$ and extend it to the take-off angle $\phi$ and out-of-plane curvature $\tau_{y y}$. We decompose $\phi$ into two additive factors. One of them is the take-off angle corresponding to a constant velocity field, thus it is known analytically. And we decompose $\tau_{y y}$ into two multiplicative factors. One of them is the out-of-plane curvature corresponding to a constant velocity field, thus it is known analytically too. The factorization of $\phi$ or $\tau_{y y}$ results in an underlying function that satisfies a factored advection equation. To solve the factored equations, we will design a third-order Lax-Friedrichs sweeping scheme based on the third-order WENO finite-difference reconstruction.

# 3 Factored eikonal and transport equations 

We first recall the factored eikonal equation in [3]. Let us consider a factored decomposition

$$
\left\{\begin{array}{l}
\tau(x, z)=\tau_{0}(x, z) u(x, z) \\
s(x, z)=s_{0}(x, z) \alpha(x, z)
\end{array}\right.
$$

and assume that $\tau_{0}$ satisfies

$$
\left|\nabla \tau_{0}\right|=s_{0}
$$

with the initial condition

$$
\lim _{(x, z) \rightarrow\left(x_{0}, z_{0}\right)}\left(\frac{\tau_{0}(x, z)}{\sqrt{\left(x-x_{0}\right)^{2}+\left(z-z_{0}\right)^{2}}}-s_{0}(x, z)\right)=0
$$

We choose $s_{0}$ as some constant, thus

$$
\tau_{0}=\frac{\sqrt{\left(x-x_{0}\right)^{2}+\left(z-z_{0}\right)^{2}}}{v_{0}}
$$

is the traveltime corresponding to the constant velocity field $v_{0}=1 / s_{0}$.
The function substitution transforms the eikonal equation (1) into the factored eikonal equation

$$
\sqrt{\tau_{0}^{2}\left(u_{x}^{2}+u_{z}^{2}\right)+2 \tau_{0} u\left(\tau_{0 x} u_{x}+\tau_{0 z} u_{z}\right)+u^{2} s_{0}^{2}}=s
$$

The factor $\tau_{0}$ captures the source singularity such that the underlying function $u$ is smooth in a neighborhood of the source.

For the constant velocity $v_{0}$, the take-off angle in the homogeneous medium, denoted as $\phi_{0}$, is constant along any ray

$$
\nabla \tau_{0} \cdot \nabla \phi_{0}=0
$$

Thus substituting the following decomposition

$$
\phi(x, z)=\phi_{0}(x, z)+\psi(x, z)
$$

into equation (4) and using equations (8) and (12), we get a factored advection equation

$$
\nabla \psi \cdot\left(\nabla \tau_{0} u+\tau_{0} \nabla u\right)+\tau_{0} \nabla u \cdot \nabla \phi_{0}=0
$$

Because $\phi_{0}$ is known analytically and captures the local properties of $\phi$, the underlying additive factor $\psi$ can be viewed as a small perturbation to $\phi_{0}$ locally at the source.

For the constant velocity $v_{0}$, the out-of-plane curvature in the homogeneous medium, denoted as $\tau_{y y 0}$, satisfies the following advection equation

$$
\frac{\partial \tau_{0}}{\partial x} \frac{\partial \tau_{y y 0}}{\partial x}+\frac{\partial \tau_{0}}{\partial z} \frac{\partial \tau_{y y 0}}{\partial z}+\tau_{y y 0}^{2}=0
$$

Considering the decomposition

$$
\tau_{y y}(x, z)=\tau_{y y 0}(x, z) c(x, z)
$$

we substitute it into (7) with the help of equations (8) and (15) to get another factored advection equation

$$
\left(\tau_{y y 0} \tau_{0} \nabla u+\tau_{y y 0} u \nabla \tau_{0}\right) \cdot \nabla c+\left(\tau_{0} \nabla \tau_{y y 0} \cdot \nabla u-\tau_{y y 0}^{2} u\right) c+\tau_{y y 0}^{2} c^{2}=0
$$

Since $\tau_{y y 0}$ is known analytically and captures the source singularity, the underlying factor $c$ is smooth in a neighborhood of the source.

With the decomposition (8) and (13), we have

$$
\left\{\begin{array}{l}
\nabla \tau=\tau_{0} \nabla u+u \nabla \tau_{0} \\
\nabla \phi=\nabla \phi_{0}+\nabla \psi
\end{array}\right.
$$

In order to get $\nabla \tau, \nabla \phi$ and $\tau_{y y}$, we need to compute $u, \nabla u, \psi, \nabla \psi$ and c. Thus we need to solve the factored eikonal equation (11) and the factored advection equations (14) and (17). The traveltime $\tau_{0}$, take-off angle $\phi_{0}$ and out-of-plane curvature $\tau_{y y 0}$ corresponding to the constant velocity field $v_{0}$ capture the source singularity properly so that the underlying functions $u, \psi$ and $c$ are smooth near the source. Consequently, we need not worry about the upwind singularity at the source when solving the factored eikonal and advection equations, and it is relatively easy to design high-order schemes for solving (11), (14) and (17) so that we cab compute the underlying functions $u, \psi$ and $c$ with high accuracy. Once $u, \psi$ and $c$ are available, we can compute the amplitude with formulas (5) or (6).

# 4 Third-order accurate Lax-Friedrichs scheme 

We present a Lax-Friedrichs scheme for the factored equations (11), (14) and (17) on a rectangular mesh $\Omega^{h}$ with grid size $h$ which covers the domain $\Omega$. Let us consider the following generic equation

$$
H\left(x, z, u, u_{x}, u_{z}\right)=f(x, z)
$$

where $H$ is a given Hamiltonian.
At a grid point $(i, j)=\left(x_{i}, z_{j}\right)$ with neighbors

$$
N\{i, j\}=\left\{\left(x_{i-1}, z_{j}\right),\left(x_{i+1}, z_{j}\right),\left(x_{i}, z_{j-1}\right),\left(x_{i}, z_{j+1}\right)\right\}
$$

we consider the Lax-Friedrichs numerical Hamiltonian to approximate $H[9,5]$

$$
\begin{aligned}
& H^{L F}\left(x_{i}, z_{j}, u_{i, j}, u_{N\{i, j\}}\right)=H\left(x_{i}, z_{j}, u_{i, j}, \frac{u_{i+1, j}-u_{i-1, j}}{2 h}, \frac{u_{i, j+1}-u_{i, j-1}}{2 h}\right) \\
& \quad-\alpha_{x} \frac{u_{i+1, j}-2 u_{i, j}+u_{i-1, j}}{2 h}-\alpha_{z} \frac{u_{i, j+1}-2 u_{i, j}+u_{i, j-1}}{2 h}
\end{aligned}
$$

where $\alpha_{x}$ and $\alpha_{z}$ are chosen such that, for fixed $\left(x_{i}, z_{j}\right)$,

$$
\begin{aligned}
& \frac{\partial H^{L F}}{\partial u_{i, j}}=\frac{\partial H}{\partial u_{i, j}}\left(x_{i}, z_{j}, u_{i, j}, \frac{u_{i+1, j}-u_{i-1, j}}{2 h}, \frac{u_{i, j+1}-u_{i, j-1}}{2 h}\right)+\frac{\alpha_{x}+\alpha_{z}}{h} \geq 0 \\
& \frac{\partial H^{L F}}{\partial u_{i+1, j}}=\frac{1}{2 h} H_{1}\left(x_{i}, z_{j}, u_{i, j}, \frac{u_{i+1, j}-u_{i-1, j}}{2 h}, \frac{u_{i, j+1}-u_{i, j-1}}{2 h}\right)-\frac{\alpha_{x}}{h} \leq 0 \\
& \frac{\partial H^{L F}}{\partial u_{i-1, j}}=-\frac{1}{2 h} H_{1}\left(x_{i}, z_{j}, u_{i, j}, \frac{u_{i+1, j}-u_{i-1, j}}{2 h}, \frac{u_{i, j+1}-u_{i, j-1}}{2 h}\right)-\frac{\alpha_{x}}{h} \leq 0 \\
& \frac{\partial H^{L F}}{\partial u_{i, j+1}}=\frac{1}{2 h} H_{2}\left(x_{i}, z_{j}, u_{i, j}, \frac{u_{i+1, j}-u_{i-1, j}}{2 h}, \frac{u_{i, j+1}-u_{i, j-1}}{2 h}\right)-\frac{\alpha_{z}}{h} \leq 0 \\
& \frac{\partial H^{L F}}{\partial u_{i, j-1}}=-\frac{1}{2 h} H_{2}\left(x_{i}, z_{j}, u_{i, j}, \frac{u_{i+1, j}-u_{i-1, j}}{2 h}, \frac{u_{i, j+1}-u_{i, j-1}}{2 h}\right)-\frac{\alpha_{z}}{h} \leq 0
\end{aligned}
$$

$H_{1}$ and $H_{2}$ denote the derivatives of $H$ with respect to the first and second gradient variable, respectively. For example, we can choose

$$
\begin{aligned}
& \alpha_{x}=\max _{m \leq u \leq M, A \leq p \leq B, C \leq q \leq D}\left\{\frac{1}{2}\left|H_{1}(x, z, u, p, q)\right|+\left|\frac{\partial H}{\partial u}(x, z, u, p, q)\right|\right\} \\
& \alpha_{z}=\max _{m \leq u \leq M, A \leq p \leq B, C \leq q \leq D}\left\{\frac{1}{2}\left|H_{2}(x, z, u, p, q)\right|+\left|\frac{\partial H}{\partial u}(x, z, u, p, q)\right|\right\}
\end{aligned}
$$

The numerical Hamiltonian $H^{L F}$ is monotone for $m \leq u_{i, j} \leq M, A \leq p \leq B$ and $C \leq q \leq D$ with $p=\left(u_{i+1, j}-u_{i-1, j}\right) / 2 h$ and $q=\left(u_{i, j+1}-u_{i, j-1}\right) / 2 h$.

Then we have a first-order Lax-Friedrichs scheme

$$
\begin{aligned}
& u_{i, j}^{n e w}= \\
& \left(\frac{1}{\alpha_{x} / h+\alpha_{z} / h}\right)\left[f_{i, j}-H\left(x_{i}, z_{j}, u_{i, j}^{o l d}, \frac{u_{i+1, j}-u_{i-1, j}}{2 h}, \frac{u_{i, j+1}-u_{i, j-1}}{2 h}\right)\right. \\
& \left.+\alpha_{x} \frac{u_{i+1, j}+u_{i-1, j}}{2 h}+\alpha_{z} \frac{u_{i, j+1}+u_{i, j-1}}{2 h}\right]
\end{aligned}
$$

To design high-order sweeping schemes, we follow the strategy in [21] to replace $u_{i-1, j}, u_{i+1, j}, u_{i, j-1}$ and $u_{i, j+1}$ with

$$
\begin{array}{ll}
u_{i-1, j}=u_{i, j}-h\left(u_{x}\right)_{i, j}^{-}, & u_{i+1, j}=u_{i, j}+h\left(u_{x}\right)_{i, j}^{+} \\
u_{i, j-1}=u_{i, j}-h\left(u_{z}\right)_{i, j}^{-}, & u_{i, j+1}=u_{i, j}+h\left(u_{z}\right)_{i, j}^{+}
\end{array}
$$

$\left(u_{x}\right)_{i, j}^{-}$and $\left(u_{x}\right)_{i, j}^{+}$are third-order upwind-biased WENO approximations of $u_{x}$; $\left(u_{z}\right)_{i, j}^{-}$and $\left(u_{z}\right)_{i, j}^{+}$are third-order upwind-biased WENO approximations of $u_{z}$. That is,

$$
\left(u_{x}\right)_{i, j}^{-}=\left(1-\omega_{-}\right)\left(\frac{u_{i+1, j}-u_{i-1, j}}{2 h}\right)+\omega_{-}\left(\frac{3 u_{i, j}-4 u_{i-1, j}+u_{i-2, j}}{2 h}\right)
$$

with

$$
\omega_{-}=\frac{1}{1+2 \gamma_{-}^{2}}, \quad \gamma_{-}=\frac{\epsilon+\left(u_{i, j}-2 u_{i-1, j}+u_{i-2, j}\right)^{2}}{\epsilon+\left(u_{i+1, j}-2 u_{i, j}+u_{i-1, j}\right)^{2}}
$$

and

$$
\left(u_{x}\right)_{i, j}^{+}=\left(1-\omega_{+}\right)\left(\frac{u_{i+1, j}-u_{i-1, j}}{2 h}\right)+\omega_{+}\left(\frac{-3 u_{i, j}+4 u_{i+1, j}-u_{i+2, j}}{2 h}\right)
$$

with

$$
\omega_{+}=\frac{1}{1+2 \gamma_{+}^{2}}, \quad \gamma_{+}=\frac{\epsilon+\left(u_{i, j}-2 u_{i+1, j}+u_{i+2, j}\right)^{2}}{\epsilon+\left(u_{i+1, j}-2 u_{i, j}+u_{i-1, j}\right)^{2}}
$$

Similarly, we can define third-order WENO approximations for $\left(u_{z}\right)_{i, j}^{-}$and $\left(u_{z}\right)_{i, j}^{+}$. $\epsilon$ is a small positive number to avoid division by zero.

Thus we have a Lax-Friedrichs scheme based on the third-order WENO approximations,

$$
\begin{aligned}
& u_{i, j}^{n e w}= \\
& \left(\frac{1}{\alpha_{x} / h+\alpha_{z} / h}\right)\left[f_{i, j}-H\left(x_{i}, z_{j}, u_{i, j}^{o l d}, \frac{\left(u_{x}\right)_{i, j}^{-}+\left(u_{x}\right)_{i, j}^{+}}{2}, \frac{\left(u_{z}\right)_{i, j}^{-}+\left(u_{z}\right)_{i, j}^{+}}{2}\right)\right. \\
& \left.+\alpha_{x} \frac{2 u_{i, j}^{o l d}+h\left(\left(u_{x}\right)_{i, j}^{+}-\left(u_{x}\right)_{i, j}^{-}\right)}{2 h}+\alpha_{z} \frac{2 u_{i, j}^{o l d}+h\left(\left(u_{z}\right)_{i, j}^{+}-\left(u_{z}\right)_{i, j}^{-}\right)}{2 h}\right]
\end{aligned}
$$

$u_{i, j}^{n e w}$ denotes the to-be-updated numerical solution for $u$ at the grid point $(i, j)$, and $u_{i, j}^{o l d}$ denotes the current old value for $u$ at the same point.

The third-order Lax-Friedrichs sweeping method for equation (19) is summarized as follows $[5,21]:$

1 Initialization: assign exact values or interpolate values at grid points within a square region centered at the source point with size equal to $2 h \times 2 h$, such that the grid points are enough for the third-order WENO approximations. These values are fixed during iterations.

2 Iterations: update $u_{i, j}^{n e w}$ in (29) by Gauss-Seidel iterations with four alternating directions:
(1) $i=1: I, j=1: J$
(2) $i=1: I, j=J: 1$;
(3) $i=I: 1, j=1: J$;
(4) $i=I: 1, j=J: 1$.

3 Convergence: if

$$
\left|u_{i, j}^{n e w}-u_{i, j}^{o l d}\right|_{\infty} \leq \delta
$$

where $\delta$ is a given convergence threshold value, the iterations converge and the algorithm stops.

We use this scheme to solve the factored equations (without confusion of notations):

- Equation (11) with Hamiltonian and $f$ as

$$
\begin{aligned}
& H\left(x, z, u, u_{x}, u_{z}\right)=\sqrt{\tau_{0}^{2}\left(u_{x}^{2}+u_{z}^{2}\right)+2 \tau_{0} u\left(\tau_{0_{x}} u_{x}+\tau_{0_{z}} u_{z}\right)+u^{2} s_{0}^{2}} \\
& f=s
\end{aligned}
$$

- Equation (14) with Hamiltonian and $f$ as

$$
\begin{aligned}
& H\left(x, z, \phi, \phi_{x}, \phi_{z}\right)=\left(\tau_{0_{x}} u+\tau_{0} u_{x}\right) \phi_{x}+\left(\tau_{0_{z}} u+\tau_{0} u_{z}\right) \phi_{z}+\tau_{0}\left(u_{x} \phi_{0_{x}}+u_{z} \phi_{0_{z}}\right) \\
& f=0
\end{aligned}
$$

- Equation (17) with Hamiltonian and $f$ as

$$
\begin{aligned}
& H\left(x, z, c, c_{x}, c_{z}\right)=\left(\tau_{0} \tau_{y y 0} u_{x}+u \tau_{y y 0} \tau_{0 x}\right) c_{x}+\left(\tau_{0} \tau_{y y 0} u_{z}+u \tau_{y y 0} \tau_{0 z}\right) c_{z} \\
& \quad+\left[\tau_{0}\left(\tau_{y y 0_{x}} u_{x}+\tau_{y y 0_{z}} u_{z}\right)-\tau_{y y 0}^{2} u\right] c+\tau_{y y 0}^{2} c^{2}
\end{aligned}
$$

$f=0$.

# 5 Numerical Examples 

In this section, we present several examples to demonstrate our method.

# 5.1 Traveltime and amplitude 

To justify our numerical schemes, we first use an example to compare our results with those obtained by the adaptive method in [11]. Then we apply our method to three other velocity models including the smooth Marmousi velocity model.

Example 1: in this example, we consider a velocity field given by

$$
v(x, z)= \begin{cases}1.0, & \text { if } z \leq 0.18 \\ 1.0+0.25 \sin (x+1.1)(z-0.18)^{2}, & \text { else }\end{cases}
$$

The domain is $[-1,1] \times[0,3]$. The source is located at $(0,0)$. The velocity field and the traveltime computed by our method are shown in Figure 1. We compare the numerical results obtained by our method with those by the adaptive method in [11]. Figure 2 shows the comparisons for $\tau_{x}, \tau_{z}, \phi_{x}, \phi_{z}, \tau_{y y}$ and $2.5-\mathrm{D}$ amplitude on a $101 \times 151$ mesh. From the figure we see that numerical results computed by our method match with the results obtained by the adaptive method.
![img-0.jpeg](img-0.jpeg)

Figure 1: Example 1. left: velocity field; right: traveltime with our method.
Example 2: in this example, we consider a velocity field in [3] given by

$$
v(x, z)=\frac{1}{\sqrt{4.0+2.0\left[g_{x}\left(x-x_{0}\right)+g_{z}\left(z-z_{0}\right)\right]}}
$$

with $\left(g_{x}, g_{z}\right)=(0,-3)$ and the source point $\left(x_{0}, z_{0}\right)=(0.25,0)$. In this case, the exact traveltime is known which is smooth. The domain is $[0,0.5] \times[-0.25,0.5]$. Table 5.1 shows the maximum error and $L_{1}$ error (with magnitude $10^{-7}$ ) in an interior region of the domain. Figure 3 shows the velocity field and the traveltime computed by our method. Figure 4 shows the results for $\tau_{x}, \tau_{z}, \phi_{x}, \phi_{z}, \tau_{y y}$ and $2.5-\mathrm{D}$ amplitude on a $201 \times 301$ mesh. For illustration purpose, we only show the contours for $z>0.025$.

![img-1.jpeg](img-1.jpeg)

Figure 2: Example 1. blue line: our method. red dash line: adaptive method. Top: τ_{x}, φ_{x} and τ_{yy}; Bottom: τ_{z}, φ_{z} and 2.5-D amplitude. Zoom-in to the window on the right.

| Error of traveltime (1.0e-7) |  |  |  |  |
| :-- | :-- | :-- | :-- | :-- |
| Mesh | 51 × 76 | 101 × 151 | 201 × 301 | 401 × 601 |
| Maximum error | 229.09 | 35.33 | 1.5155 | 0.007642 |
| L1 error | 1.163 | 0.0921 | 0.003124 | 0.00021 |

Table 1: Example 2. Maximum error and L1 error of the traveltime.

Example 3 (Sinusoidal model): in this example, the velocity field is given by

$$v(x, z) = 1 + 0.2 \sin(0.5\pi z) \sin(3\pi(x + 0.55)).$$

The domain is [-1, 1] × [0, 2] and the source point is (0, 0). Figure 5 shows the velocity field and the traveltime computed by our method. Figure 6 shows the results for τ_{x}, τ_{z}, φ_{x}, φ_{z}, τ_{yy} and 2.5-D amplitude on a 201 × 201 mesh.

As is known, in this case the traveltime field is not smooth everywhere away from the source. In fact, the physical traveltime field is multivalued as shown in [10]. However, the high-order Lax-Friedrichs sweeping scheme is based on the monotone scheme which only yields the viscosity-solution based single-valued solutions. Consequently, we see kinks in the computed traveltime field as shown in Figure 5; when numerically differentiated these kinks will produce discontinuities as shown in Figure 6. Nevertheless, those discontinuities are

![img-2.jpeg](img-2.jpeg)

Figure 3: Example 2. left: velocity field; right: traveltime with our method.
confined near the kinks because the underlying numerical schemes are essentially upwinding.

Example 4 (Marmousi velocity model): in this example, we consider the smooth Marmousi velocity model as in figure 7. Note that the velocity is rescaled by a factor $10^{-4}$. The traveltime computed with our method is shown in figure 7. Figure 8 shows the results for $\tau_{x}, \tau_{z}, \phi_{x}, \phi_{z}, \tau_{y y}$ and 2.5-D amplitude.

In this case, we also see kinks in computed traveltime field and discontinuities in other computed quantities.

# 5.2 Wavefield construction 

Now that the traveltime and the amplitude are available, we may construct the asymptotic Green function for the Helmholtz equation in the high frequency regime. However, because computed traveltimes and amplitudes are based on the concept of viscosity solutions, the constructed Green function is an approximation to the true Green function in the weak sense in that the constructed Green function approximates the true Green function faithfully only when the traveltime field is smooth (with no kinks), and the constructed Green function approximates the true Green function unfaithfully when the traveltime field is not smooth with kinks. To demonstrate this feature clearly, we will show a couple of examples.

Example 5 (Green functions): we use our results to approximate the Green functions for the Helmholtz equation in the high frequency regime,

$$
\nabla^{2} G(x, z, \omega)+\frac{\omega^{2}}{v^{2}(x, z)} G(x, z, \omega)=-\delta\left(x-x_{0}\right) \delta\left(z-z_{0}\right)
$$

where $G(x, z, \omega)$ is the Green function.

![img-3.jpeg](img-3.jpeg)

Figure 4: Example 2. Top: $\tau_{x}, \phi_{x}$ and $\tau_{y y}$; Bottom: $\tau_{z}, \phi_{z}$ and 2.5-D amplitude.

We approximate the two-dimensional Green function in the WKBJ form (Appendix C, [7]),

$$
G_{2}\left(x, z, \omega \approx \frac{1}{\sqrt{\omega}} A(x, z) e^{i(\omega \tau(x, z)+\frac{\pi}{4})}\right.
$$

where $A$ is given by equation (5).
Two velocity models are used to test our results, and we compare the constructed Green functions with those obtained by the direct solver of the Helmholtz equation in [2]. We choose $\omega=32 \pi$.

1. $v(x, z) \equiv 5.0,\left(x_{0}, z_{0}\right)=(0.5,0.5)$, domain $[0,1] \times[0,1]$. Figure 9 shows the results for the two-dimensional Green function on a $1200 \times 1200$ mesh. The results by our method are very close to those obtained by the Helmholtz solver.
x
2. $v(x, z)=1+0.2 \sin (0.5 \pi z) \sin (3 \pi(x+0.05)),\left(x_{0}, z_{0}\right)=(0.5,0.1)$, domain $[0,1] \times[0,2]$. Figure 10 shows the results for the real part of the twodimensional Green function on a $1600 \times 800$ mesh. Figure 11 shows two slices at $z=0.3$ (no kinks, no caustics) and $z=1.5$ (kinks, caustics).

![img-4.jpeg](img-4.jpeg)

Figure 5: Example 4. left: velocity field; right: traveltime with our method.

Case 1. Because the traveltime field is smooth everywhere away from the source, the constructed asymptotic Green function approximates the true Green function faithfully.

Case 2. Because the traveltime field is not smooth, the constructed Green function in the weak sense cannot approximate the true Green function faithfully as shown in Figure 10. In fact, the traveltime field in the viscosity-solution sense is single-valued, and the resulting ray structure is shown in the bottom-left subfigure. On the other hand, to reconstruct the true Green function, we need the multivalued traveltime field, and the resulting ray structure is shown in the bottom-right subfigure. Consequently, there is an essential difference between single-valued and multivalued traveltime fields.

We also mention that before kinks appear in the single-valued traveltime field or caustics appear in the multivalued traveltime field, the true traveltime field is smooth and the asymptotic Green function in the single-valued and multivalued sense approximates the true Green function faithfully. Only after kinks appear in the single-valued traveltime field or caustics appear in the multivalued traveltime field, the two traveltime fields yield totally different Green functions, as shown clearly in Figure 11.

# 6 Conclusions 

We present a factorization technique based on the factored eikonal equation to compute the take-off angle and the out-of-plane curvature, thus the amplitude. We decompose the take-off angle and the out-of plane curvature into two additive and multiplicative factors, respectively. One of them is known analytically corresponding to a constant velocity field, and it captures the local properties of the take-off angle or the out-of-plane curvature well in the neighborhood of the source. Then a third-order WENO based Lax-Friedrichs sweeping method

![img-5.jpeg](img-5.jpeg)

Figure 6: Example 4. Top: $\tau_{x}, \phi_{x}$ and $\tau_{y y}$; Bottom: $\tau_{z}, \phi_{z}$ and 2.5-D amplitude.
is applied to solve the factored equations numerically. The advantage of decomposing the take-off angle into two additive factors is that since the known factor captures the local properties such as the angular directions of the take-off angle at the source, the other factor can be initialized easily at the source. While in the adaptive method, the take-off angle need to be initialized at more grid points so that enough angular directions can be covered. The advantage of decomposing the out-of-plane curvature into tow multiplicative factors is that since the known factor captures the source singularity, the other factor is smooth at the source. Numerical examples are presented to demonstrate the performance of our new method.

# References 

[1] M. G. Crandall and P.-L. Lions, Viscosity solutions of Hamilton-Jacobi equations, Tans. Amer. Math. Soc. 277 (1983), 1-42.
[2] Y. A. Erlangga, C. W. Oosterlee, and C. Vuik, A novel multigrid-based preconditioner for the heterogeneous Helmholtz equation, SIAM Journal on Scientific Computing 27 (2006), 1471-1492.
[3] S. Fomel, S. Luo, and H.-K. Zhao, Fast sweeping method for the factored eikonal equation, Journal of Comp. Phys. 228 (2009), no. 17, 6440-6455.

![img-6.jpeg](img-6.jpeg)

Figure 7: Example 5. left: Marmousi velocity field; right: traveltime with our method.
[4] F. Friedlander, Sound pulses, Cambridge Univ. Press, 1958.
[5] C. Y. Kao, S. Osher, and J. Qian, Lax-Friedrichs sweeping schemes for static Hamilton-Jacobi equations, Journal of Computational Physics 196 (2004), 367-391.
[6] S. Kim and R. Cook, 3-D traveltime computation using second-order ENO scheme, Geophysics 64 (1999), 1867-1876.
[7] S. Leung, J. Qian, and R. Burridge, Eulerian Gaussian beams for highfrequency wave propagation, Geophysics 72 (2007), no. 5, SM61-SM76.
[8] P.-L. Lions, Generalized solutions of Hamilton-Jacobi equations, Pitman, Boston, 1982.
[9] S. Osher and C.-W. Shu, High-order essentially nonoscillatory schemes for Hamilton-Jacobi equations, SIAM J. Math. Anal. 28 (1991), no. 4, 907-922.
[10] J. Qian and S. Leung, A level set method for paraxial multivalued traveltimes, J. Comp. Phys. 197 (2004), 711-736.
[11] J. Qian and W. W. Symes, An Adaptive Finite-Difference Method for Traveltimes and Amplitudes, Geophysics 67 (2002), 167-176.
[12] F. Qin, Y. Luo, K. B. Olsen, W. Cai, and G. T. Schuster, Finite difference solution of the eikonal equation along expanding wavefronts, Geophysics 57 (1992), 478-487.
[13] W. A. Jr. Schneider, Robust and efficient upwind finite-difference traveltime calculations in three dimensions, Geophysics 60 (1995), 1108-1117.

![img-7.jpeg](img-7.jpeg)

Figure 8: Example 5. Top: $\tau_{x}, \phi_{x}$ and $\tau_{y y}$; Bottom: $\tau_{z}, \phi_{z}$ and 2.5-D amplitude.
[14] S. Serna and J. Qian, A stopping criterion for higher-order sweeping schemes for static hamilton-jacobi equations, J. Comput. Math. 28 (2010), $552-568$.
[15] J. A. Sethian, Level Set Methods and Fast Marching Methods: Evolving Interfaces in Computational Geometry, Fluid Mechanics, Computer Vision, and Materials Science, Cambridge University Press, 1999.
[16] J. A. Sethian and A. M. Popovici, 3-D traveltime computation using the fast marching method, Geophysics 64 (1999), no. 2, 516-523.
[17] W. W. Symes, R. Versteeg, A. Sei, and Q. H. Tran, Kirchhoff simulation migration and inversion using finite-difference travel-times and amplitudes, TRIP tech. Report, Rice U. (1994).
[18] I. A. Molotkov V. Cerveny and I. Psencik, , Univ. Karlova Press, 1977.
[19] J. van Trier and W.W. Symes, Upwind finite-difference calculation of traveltimes, Geophysics 56 (1991), no. 06, 812-821.
[20] J. E. Vidale, Finite-difference calculation of traveltimes in three dimensions, Geophysics 55 (1990), no. 05, 521-526.

![img-8.jpeg](img-8.jpeg)

Figure 9: Example 6.1. Two-dimensional Green function. Top: image of the real part of the Green function (left: our method, right: Helmholtz solver). Bottom: two slices of the Green function (real part) at $x=0.3$ (left) and $z=0.3$ (right). Red circle: our method. Blue dot: Helmholtz solver.
[21] Y.-T. Zhang, H.-K. Zhao, and J. Qian, High order fast sweeping methods for static Hamilton-Jacobi equations, Journal of Scientific Computing 29 (2006), 25-56.
[22] H.-K. Zhao, A fast sweeping method for eikonal equations, Mathematics of Computation 74 (2005), 603-627.

![img-9.jpeg](img-9.jpeg)

Figure 10: Example 6.2. Top: two-dimensional Green functions (real part); Bottom: Green functions with rays. Left: our method. Right: Helmholtz solver.

![img-10.jpeg](img-10.jpeg)

Figure 11: Example 6.2. Two slices of the two-dimensional Green function at $z=0.3$ (no kinks, no caustics) and $z=1.5$ (kinks, caustics). Red circle: our method. Blue dot: Helmholtz solver.

