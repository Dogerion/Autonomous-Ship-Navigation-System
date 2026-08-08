# Linear Quadratic Programming (LQP) Formulation for Nomoto MPC

This document outlines the mathematical translation of the continuous-time Nomoto vessel kinematics and the target optimal control cost function into the standard discrete-time Quadratic Programming (QP) format required by solvers like OSQP.

## 1. System Dynamics (State-Space Representation)

The vessel's dynamics are governed by the first-order Nomoto model and the integral of the heading error:
$$ T \dot{r} + r = K \delta $$
$$ \dot{\psi} = r $$
$$ \dot{I}_{\psi} = \psi $$

We define the state vector $x$ and control vector $u$ at step $k$:
$$ x_k = \begin{bmatrix} \psi_k \\ r_k \\ I_{\psi, k} \end{bmatrix}, \quad u_k = \begin{bmatrix} \delta_k \end{bmatrix} $$

Using Euler forward integration with a time step $dt$, the continuous equations are discretized into the standard linear state-space form $x_{k+1} = A x_k + B u_k$, reflecting the exact cascading integration sequence of the environment:

$$
A = \begin{bmatrix} 
1 & dt - \frac{dt^2}{T} & 0 \\ 
0 & 1 - \frac{dt}{T} & 0 \\ 
dt & dt^2 - \frac{dt^3}{T} & 1 
\end{bmatrix}
, \quad
B = \begin{bmatrix} 
\frac{K \cdot dt^2}{T} \\ 
\frac{K \cdot dt}{T} \\ 
\frac{K \cdot dt^3}{T} 
\end{bmatrix}
$$

## 2. Objective Function

The objective of the Model Predictive Controller is to minimize the sum of the state error and control effort over a prediction horizon $N$. The penalty at a single step $k$ is defined as:
$$ \text{Cost}_k = x_k^T Q x_k + u_k^T R u_k + \Delta u_k^T R_{\text{rate}} \Delta u_k $$
Which expands in our scalar 1D control system to:
$$ \text{Cost}_k = w_1 \psi_k^2 + w_2 \delta_k^2 + w_3 (\delta_k - \delta_{k-1})^2 $$

We expand the objective function over the entire horizon:
$$ J = \sum_{k=0}^{N-1} \left( x_k^T Q x_k + u_k^T R u_k + \Delta u_k^T R_{\text{rate}} \Delta u_k \right) + x_N^T Q_N x_N $$

Where **$Q$** is the diagonal state penalty matrix. The integral error $I_{\psi}$ and raw yaw rate $r$ are unpenalized in the objective:
$$
Q = \begin{bmatrix} w_1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
$$

Where **$R$** is the control penalty matrix (penalizing raw rudder deflection), and **$R_{\text{rate}}$** is the penalty matrix for actuator speed:
$$
R = \begin{bmatrix} w_2 \end{bmatrix}, \quad R_{\text{rate}} = \begin{bmatrix} w_3 \end{bmatrix}
$$

### Terminal Cost ($Q_N$)
In highly stochastic environments (e.g., dynamic wave disturbances), heavily penalizing the final state of the horizon ($x_N$) can cause the controller to aggressively over-correct based on highly uncertain future predictions. 

To ensure the controller behaves robustly and smoothly in the presence of random noise, we do not apply an artificial terminal multiplier or use infinite-horizon approximations (like DARE). The terminal cost matrix is treated identically to the intermediate step cost matrix:
$$ Q_N = Q $$

## 3. Standard QP Cast (The $P$ and $q$ Matrices)

Standard QP solvers like OSQP require the problem to be cast into a single, massive quadratic form over the entire decision vector $z$:
$$ \text{Minimize} \quad \frac{1}{2} z^T P z + q^T z $$
$$ \text{where} \quad z = \begin{bmatrix} x_0 \\ \vdots \\ x_N \\ u_0 \\ \vdots \\ u_{N-1} \end{bmatrix} $$

### The Hessian Matrix ($P$)
The state penalties ($Q$ and $Q_N$) populate the top-left blocks of $P$. 

The control penalties populate the bottom-right block $P_u$. To account for the squared difference term $w_3 (u_k - u_{k-1})^2 = w_3 u_k^2 - 2w_3 u_k u_{k-1} + w_3 u_{k-1}^2$, we construct $P_u$ as a symmetric **tridiagonal matrix**.
Because the solver objective includes a $\frac{1}{2}$ multiplier, all weights are multiplied by $2$ when placed into $P$:

*   **Main Diagonal ($u_k^2$ terms):** $2(w_2 + 2w_3)$
*   **Final Diagonal ($u_{N-1}^2$ term):** $2(w_2 + w_3)$ *(no future step to difference against)*
*   **Off-Diagonals ($u_k u_{k-1}$ cross terms):** $-2w_3$

$$ P = \text{block\_diag}(Q, \dots, Q, Q_N, P_u) $$

### The Linear Vector ($q$)
The vector $q$ handles the linear components of the objective. For the very first control step $u_0$, the difference penalty is $w_3 (u_0 - u_{-1})^2$. 

Expanding this yields a linear term $-2w_3 u_{-1} u_0$, where $u_{-1}$ is a constant representing the physical rudder angle applied at the *previous* timestep. We inject this coupling into the slot of $q$ corresponding to $u_0$:
$$ q_{u_0} = -2 w_3 u_{-1} $$

## 4. Constraints

The solver must strictly obey two physical rules:

1.  **Dynamics (Equality Constraints):** The predicted states must follow the physical transition matrices.
    $$ x_{k+1} - A x_k - B u_k = 0 $$
    $$ x_0 = \text{current\_state} $$

2.  **Actuator Limits (Inequality Constraints):** The rudder angle cannot exceed the mechanical hardware limits of the vessel.
    $$ -\delta_{\max} \leq u_k \leq \delta_{\max} $$

These are stacked vertically into large sparse matrices ($A_{\text{eq}}$, $A_{\text{ineq}}$) and passed to OSQP to bound the optimal trajectory search space.