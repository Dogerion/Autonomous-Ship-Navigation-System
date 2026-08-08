# The First-Order Nomoto Model

This document provides a mathematical and physical overview of the first-order Nomoto model, which serves as the core physical engine for the vessel kinematics in the `NomotoEnv` simulation.

## 1. Physical Background

Maneuvering a large surface vessel is a highly complex hydrodynamic process involving 3 degrees of freedom (surge, sway, and yaw) and non-linear fluid damping.

In 1957, K. Nomoto demonstrated that for standard course-keeping and steering tasks where the ship is moving at a relatively constant forward speed (surge), the complex non-linear equations can be simplified into a highly accurate linear transfer function relating the **rudder angle ($\delta$)** to the **yaw rate ($r$)**. 

This simplification is universally known as the First-Order Nomoto Model and is the global standard for autopilot design.

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

## 3. Understanding the Parameters ($K$ and $T$)

The entire dynamic profile of the ship is governed by the two parameters $K$ and $T$. In our reinforcement learning setup, these parameters are strictly hidden from the agent and randomized via Domain Randomization to force the agent to learn an adaptive control policy.

### The Turning Gain ($K$)
$K$ represents the steady-state turning ability of the vessel. 
If the rudder is held at a constant angle $\delta_{ss}$, the yaw acceleration eventually becomes zero ($\dot{r} = 0$). The equation simplifies to:
$$ r_{ss} = K \delta_{ss} $$

A high $K$ value means the ship is highly responsive to the rudder and will turn very sharply (e.g., a small patrol boat). A low $K$ value means the ship resists turning even at maximum rudder deflection (e.g., a directionally stable cargo ship).

### The Time Constant ($T$)
$T$ represents the rotational inertia (or "sluggishness") of the vessel. It dictates the transient response of the ship—how long it takes to reach that steady-state turning speed $r_{ss}$.

Physically, if the rudder is instantly deflected, it takes $T$ seconds for the ship's yaw rate to reach roughly $63.2\%$ of its final steady-state yaw rate.

A massive oil tanker with a huge mass and immense hydrodynamic added-mass will have a very large $T$ (e.g., 20+ seconds), meaning it takes a long time to start turning, and a long time to stop turning once the rudder is centered.

## 4. Discrete Time Integration (Euler Method)

Because our environment runs inside a computer simulation utilizing discrete steps, the continuous differential equations must be solved numerically. We use the Euler Forward Integration method with a time step of $dt$.

To find the yaw rate at the next timestep $t+1$:
$$ \dot{r}(t) = \frac{K \delta(t) - r(t)}{T} $$
$$ r(t+1) = r(t) + \dot{r}(t) \cdot dt $$

To find the new heading:
$$ \psi(t+1) = \psi(t) + r(t+1) \cdot dt $$

This discrete formulation forms the mathematical baseline of the `NomotoEnv.step()` function and the discrete state-space transition matrices used in the MPC baseline.
