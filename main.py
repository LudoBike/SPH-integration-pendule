import numpy as np
import matplotlib.pyplot as plt
import values as v
from numerical_scheme import simplectic_scheme, explicit_scheme, implicit_scheme


# ------------------------------------------------------------
#   1. Phase diagram
# ------------------------------------------------------------
dt = v.T0 / 50  # Timestep [s]
N = int(np.ceil(3 * v.T0 / dt))  # Number of iteration, we want to compute 3 periods

# Theorical solution
phi = np.arcsin(-v.theta_dot0 / (v.theta0 * v.omega0))
timerange = np.linspace(0, N * dt, N + 1)
theo_theta = v.theta0 * np.cos(v.omega0 * timerange + phi)
theo_theta_dot = -v.omega0 * v.theta0 * np.sin(v.omega0 * timerange + phi)

# Numerical solution
(simp_theta, simp_theta_dot) = simplectic_scheme(
    v.theta0, v.theta_dot0, v.omega0, dt, N
)
(expl_theta, expl_theta_dot) = explicit_scheme(v.theta0, v.theta_dot0, v.omega0, dt, N)
(impl_theta, impl_theta_dot) = implicit_scheme(v.theta0, v.theta_dot0, v.omega0, dt, N)

plt.plot(simp_theta, simp_theta_dot, color="red", label="Schéma simplectique")
plt.plot(expl_theta, expl_theta_dot, color="blue", label="Schéma explicite")
plt.plot(impl_theta, impl_theta_dot, color="green", label="Schéma implicite")
plt.plot(
    theo_theta,
    theo_theta_dot,
    color="black",
    linestyle=(0, (5, 5)),
    label="Résultat théorique",
)
plt.xlabel(r"$\theta(t)$")
plt.ylabel(r"$\dot{\theta}(t)$")
plt.legend()
plt.show()
