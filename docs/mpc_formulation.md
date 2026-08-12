# QP Formulation for the Nomoto MPC

This document shows how the continuous-time Nomoto model and the control cost are turned into the discrete-time Quadratic Program (QP) that OSQP solves.

## 1. System Dynamics (State-Space Representation)

The vessel's dynamics are governed by the first-order Nomoto model and the integral of the heading error:
$$ T \dot{r} + r = K \delta $$
$$ \dot{\psi} = r $$
$$ \dot{I}_{\psi} = \psi $$

We define the state vector $x$ and control vector $u$ at step $k$:
$$ x_k = \begin{bmatrix} \psi_k \\ r_k \\ I_{\psi, k} \\ \delta_{k-1} \end{bmatrix}, \quad u_k = \begin{bmatrix} v_k \end{bmatrix} $$
*(Where $v_k = \Delta \delta_k$ represents the change in rudder angle at step $k$)*

Using Euler method with a time step $dt$, the continuous equations are discretized into the standard linear state-space form $x_{k+1} = A x_k + B u_k$, reflecting the exact cascading integration sequence of the environment:

$$
A = \begin{bmatrix} 
1 & dt - \frac{dt^2}{T} & 0 & \frac{K \cdot dt^2}{T} \\ 
0 & 1 - \frac{dt}{T} & 0 & \frac{K \cdot dt}{T} \\ 
dt & dt^2 - \frac{dt^3}{T} & 1 & \frac{K \cdot dt^3}{T} \\
0 & 0 & 0 & 1
\end{bmatrix}
, \quad
B = \begin{bmatrix} 
\frac{K \cdot dt^2}{T} \\ 
\frac{K \cdot dt}{T} \\ 
\frac{K \cdot dt^3}{T} \\
1
\end{bmatrix}
$$

## 2. Objective Function

The objective of the Model Predictive Controller is to minimize the sum of the state error and control effort over a prediction horizon $N$. The penalty at a single step $k$ is defined as:
$$ \text{Cost}_k = x_k^T Q x_k + u_k^T R u_k $$
Which expands in our scalar 1D control system to:
$$ \text{Cost}_k = w_1 \psi_k^2 + w_2 \delta_{k-1}^2 + w_3 v_k^2 $$

We expand the objective function over the entire horizon:
$$ J = \sum_{k=0}^{N-1} \left( x_k^T Q x_k + u_k^T R u_k \right) + x_N^T Q_N x_N $$

Where **$Q$** is the diagonal state penalty matrix. The integral error $I_{\psi}$ and raw yaw rate $r$ are unpenalized in the objective. The penalty for physical rudder angle is applied to the 4th state ($\delta_{k-1}$):
$$
Q = \begin{bmatrix} w_1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & w_2 \end{bmatrix}
$$

Where **$R$** is the control penalty matrix (penalizing raw actuator speed $v_k = \Delta \delta_k$):
$$
R = \begin{bmatrix} w_3 \end{bmatrix}
$$

### Terminal Cost ($Q_N$)
With random wave noise, heavily penalizing the final horizon state ($x_N$) can make the controller over-correct based on uncertain future predictions.

To keep behavior smooth under noise, we do not add a terminal multiplier or an infinite-horizon approximation (like DARE). The terminal cost is the same as the per-step cost:
$$ Q_N = Q $$

## 3. Standard QP Cast (The $P$ and $q$ Matrices)

Standard QP solvers like OSQP require the problem to be cast into a single, massive quadratic form over the entire decision vector $z$:
$$ \text{Minimize} \quad \frac{1}{2} z^T P z + q^T z $$
$$ \text{where} \quad z = \begin{bmatrix} x_0 \\ \vdots \\ x_N \\ u_0 \\ \vdots \\ u_{N-1} \end{bmatrix} $$

### The Hessian Matrix ($P$)
The state penalties ($Q$ and $Q_N$) fill the top-left blocks of $P$, and the control penalty ($R$) fills the bottom-right blocks. Because the rudder-rate term is folded into the augmented state, $P$ is block-diagonal with no cross-terms.

Note that the solver objective carries a $\frac{1}{2}$ factor, so weights may need scaling by $2$ when placed into $P$. The block-diagonal structure is:

$$ 
P = \begin{bmatrix}
Q_0 & 0 & \dots & 0 & 0 & 0 & \dots & 0 \\
0 & Q_1 & \dots & 0 & 0 & 0 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & Q_N & 0 & 0 & \dots & 0 \\
0 & 0 & \dots & 0 & R_0 & 0 & \dots & 0 \\
0 & 0 & \dots & 0 & 0 & R_1 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & 0 & 0 & 0 & \dots & R_{N-1} 
\end{bmatrix}
$$
*(Where $P$ is a block diagonal matrix. For our time-invariant cost formulation, $Q_0 = Q_1 = \dots = Q_{N-1} = Q$ and $R_0 = R_1 = \dots = R_{N-1} = R$)*

### The Linear Vector ($q$)
Because the augmented state carries the previous rudder angle ($\delta_{k-1}$) without producing cross-terms in the objective, the linear objective vector is zero:
$$ q = \begin{bmatrix} 0 \\ \vdots \\ 0 \end{bmatrix} $$

## 4. Constraints

The solver must strictly obey two physical rules:

1.  **Dynamics (Equality Constraints):** The predicted states must follow the augmented physical transition matrices.
    $$ x_{k+1} - A x_k - B u_k = 0 $$
    The initial condition explicitly injects the previously executed physical rudder angle into the 4th state slot:
    $$ x_0 = \begin{bmatrix} \psi_0 \\ r_0 \\ I_{\psi, 0} \\ \delta_{\text{prev}} \end{bmatrix} $$

2.  **Actuator Limits (Inequality Constraints):** The physical rudder angle (State 4) cannot exceed the mechanical hardware limits of the vessel. We can also optionally apply inequality bounds to the rudder speed (Control $u_k$).
    $$ -\delta_{\max} \leq x_{4,k} \leq \delta_{\max} $$

These are stacked vertically into large sparse matrices ($A_{\text{eq}}$, $A_{\text{ineq}}$) and passed to OSQP to bound the optimal trajectory search space.