# The First-Order Nomoto Model

This document explains the first-order Nomoto model, which drives the ship kinematics in the `NomotoEnv` simulation.

## 1. Background

Maneuvering a surface vessel is a hydrodynamic process with 3 degrees of freedom (surge, sway, and yaw).

In 1957, K. Nomoto showed that for course-keeping and steering at a roughly constant forward speed, these equations can be reduced to a linear transfer function relating the **rudder angle ($\delta$)** to the **yaw rate ($r$)**.

This is the first-order Nomoto model, and it is a common starting point for autopilot design.

## 2. The Differential Equation

The continuous-time first-order Nomoto equation is defined as:
$$ T \dot{r}(t) + r(t) = K \delta(t) $$

Where:
*   **$r(t)$** : Yaw rate (the rotational speed of the vessel, typically in radians/second).
*   **$\dot{r}(t)$** : Yaw acceleration (radians/second$^2$).
*   **$\delta(t)$** : Commanded physical rudder angle (radians).
*   **$K$** : The ship's turning gain (1/seconds).
*   **$T$** : The ship's time constant or inertia (seconds).

The heading of the vessel, **$\psi(t)$**, is simply the integral of the yaw rate:
$$ \dot{\psi}(t) = r(t) $$

## 3. Ship Dynamic Parameters

The ship's dynamics are set by the two parameters $K$ and $T$. In this project they are hidden from the controller and randomized each episode, so the controller has to adapt rather than memorize one ship.

Both parameters fall out of the ship's yaw equation of motion. For a simplified yaw-only model:
$$ (I_z - N_{\dot r})\,\dot r = N_r\,r + N_\delta\,\delta $$
Matching this to the Nomoto form $T\dot r + r = K\delta$ gives the expressions below.

### The Turning Gain ($K$)
$K$ represents the turning ability of the vessel. 
If the rudder is held at a constant angle $\delta_{ss}$, the yaw acceleration eventually becomes zero ($\dot{r} = 0$). The equation simplifies to:
$$ r_{ss} = K \delta_{ss} $$

From the yaw equation of motion, $K$ is the rudder's yaw moment over the yaw damping:
$$ K = \frac{N_\delta}{-N_r} $$
where $N_\delta$ is the yaw moment produced per unit rudder angle. In non-dimensional form it scales with ship length $L$ and forward speed $U$ as:
$$ K = K'\,\frac{U}{L} $$

A high $K$ value means the ship is highly responsive to the rudder and will turn very sharply. A low $K$ value means the ship resists turning even at maximum rudder deflection.

### Rotational Inertia ($T$)
$T$ measures the vessel's rotational inertia. It sets how quickly the ship responds to the rudder. Which indicates how long the yaw rate takes to build up to its steady-state value $r_{ss}$.

From the same yaw equation of motion, $T$ is the ship's effective rotational inertia over its yaw damping:
$$ T = \frac{I_z - N_{\dot r}}{-N_r} $$
where $I_z - N_{\dot r}$ is the hull inertia plus hydrodynamic added inertia. In non-dimensional form it scales with ship length $L$ and forward speed $U$ as:
$$ T = T'\,\frac{L}{U} $$

A high $T$ value means the ship is sluggish — a massive oil tanker takes a long time to start turning, and a long time to stop once the rudder is centered. A low $T$ value means the ship is nimble — a light patrol boat reaches its steady-state turn rate almost immediately.

## 4. Discrete Time Integration (Euler Method)

Because our environment runs inside a computer simulation utilizing discrete steps, the continuous differential equations must be solved numerically. We use the Euler Forward Integration method with a time step of $dt$.

To find the yaw rate at the next timestep $t+1$:
$$ \dot{r}(t) = \frac{K \delta(t) - r(t)}{T} $$
$$ r(t+1) = r(t) + \dot{r}(t) \cdot dt $$

To find the new heading:
$$ \psi(t+1) = \psi(t) + r(t+1) \cdot dt $$

This discrete formulation forms the mathematical baseline of the `NomotoEnv` function and the discrete state-space transition matrices used in the MPC baseline.
