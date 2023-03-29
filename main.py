import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import values as v
from numerical_scheme import simplectic_scheme, explicit_scheme, implicit_scheme


# ------------------------------------------------------------
#   Figures parameters
# ------------------------------------------------------------
figure_path = "figures/"  # Path where to save figures
figure_dpi = 400
figure_format = "png"

try:
    os.mkdir(figure_path)
except FileExistsError:
    pass

# ------------------------------------------------------------
#   1. Phase diagram
# ------------------------------------------------------------
dt = v.T0 / 100  # Timestep [s]
N = int(np.ceil(3 * v.T0 / dt))  # Number of iteration, we want to compute 3 periods
timerange = np.linspace(0, N * dt, N + 1)

# Theorical solution
phi = np.arcsin(-v.theta_dot0 / (v.theta0 * v.omega0))
theo_theta = v.theta0 * np.cos(v.omega0 * timerange + phi)
theo_theta_dot = -v.omega0 * v.theta0 * np.sin(v.omega0 * timerange + phi)

# Numerical solution
(simp_theta, simp_theta_dot) = simplectic_scheme(
    v.theta0, v.theta_dot0, v.omega0, dt, N
)
(expl_theta, expl_theta_dot) = explicit_scheme(v.theta0, v.theta_dot0, v.omega0, dt, N)
(impl_theta, impl_theta_dot) = implicit_scheme(v.theta0, v.theta_dot0, v.omega0, dt, N)

plt.plot(
    simp_theta / v.theta0,
    simp_theta_dot / (v.omega0 * v.theta0),
    color="red",
    label="Schéma simplectique",
)
plt.plot(
    expl_theta / v.theta0,
    expl_theta_dot / (v.omega0 * v.theta0),
    color="blue",
    label="Schéma explicite",
)
plt.plot(
    impl_theta / v.theta0,
    impl_theta_dot / (v.omega0 * v.theta0),
    color="green",
    label="Schéma implicite",
)
plt.plot(
    theo_theta / v.theta0,
    theo_theta_dot / (v.omega0 * v.theta0),
    color="black",
    linestyle=(0, (3, 3)),
    label="Résultat théorique",
)
plt.xlabel(r"$\theta(t)/\theta_0$")
plt.ylabel(r"$\dot{\theta}(t)/\omega_0\theta_0$")
plt.legend()
plt.savefig(figure_path + "diagrame_phase", dpi=figure_dpi, format=figure_format)
plt.clf()


# ------------------------------------------------------------
#   2. Hamiltonian
# ------------------------------------------------------------
def hamiltonian(theta, theta_dot):
    """Return the hamiltonien of a pendulum in the state defined by (theta, theta_dot)"""
    return 0.5 * v.m * v.l**2 * (theta_dot**2 + v.omega0**2 * theta**2)


theo_hamiltonian = 0.5 * v.m * v.g * v.l * v.theta0**2 * np.ones_like(timerange)
simp_hamiltonian = hamiltonian(simp_theta, simp_theta_dot)
expl_hamiltonian = hamiltonian(expl_theta, expl_theta_dot)
impl_hamiltonian = hamiltonian(impl_theta, impl_theta_dot)

H_0 = theo_hamiltonian[0]
plt.plot(
    timerange / v.T0,
    theo_hamiltonian / H_0,
    color="black",
    linestyle=(0, (5, 5)),
    label="Résultat théorique",
)
plt.plot(
    timerange / v.T0,
    simp_hamiltonian / H_0,
    color="red",
    label="Schéma simplectique",
)
plt.plot(
    timerange / v.T0,
    expl_hamiltonian / H_0,
    color="blue",
    label="Schéma explicite",
)
plt.plot(
    timerange / v.T0,
    impl_hamiltonian / H_0,
    color="green",
    label="Schéma implicite",
)
plt.xlabel("Temps $t/T_0$")
plt.ylabel("Hamiltonien $H(t)/H(0)$")
plt.legend()
plt.savefig(figure_path + "hamiltonien", dpi=figure_dpi, format=figure_format)
plt.clf()


# ------------------------------------------------------------
#   3. Error estimation
# ------------------------------------------------------------
def error_L2_norm(num_theta, num_theta_dot, theo_theta, theo_theta_dot):
    """
    Compute the numerical error with the L2 norm
    num_theta and num_theta_dot are the numerical results
    theo_theta and theo_theta_dot are the analytical results
    """
    N = len(num_theta)  # Number of iterations
    error = np.sum(
        ((num_theta - theo_theta) / v.theta0) ** 2
        + ((num_theta_dot - theo_theta_dot) / (v.omega0 * v.theta0)) ** 2
    )
    error = np.sqrt(error / N)
    return error


dt_range = v.T0 / np.array(
    [
        1000000,
        300000,
        100000,
        30000,
        10000,
        3000,
        1000,
        300,
        100,
        50,
        30,
        20,
        15,
        13,
        11,
        10,
    ]
)
errors = np.zeros_like(dt_range)
for i in range(len(dt_range)):
    dt = dt_range[i]
    N = int(np.ceil(5 * v.T0 / dt))  # Number of iteration, we want to compute 5 periods
    timerange = np.linspace(0, N * dt, N + 1)

    # Theorical solution
    phi = np.arcsin(-v.theta_dot0 / (v.theta0 * v.omega0))
    theo_theta = v.theta0 * np.cos(v.omega0 * timerange + phi)
    theo_theta_dot = -v.omega0 * v.theta0 * np.sin(v.omega0 * timerange + phi)

    # Numerical solution with simplectic scheme
    (simp_theta, simp_theta_dot) = simplectic_scheme(
        v.theta0, v.theta_dot0, v.omega0, dt, N
    )

    errors[i] = error_L2_norm(simp_theta, simp_theta_dot, theo_theta, theo_theta_dot)

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    np.log10(dt_range[:8] / v.T0), np.log10(errors[:8])
)
print("Pente de la régression :", slope)
print("R^2 :", r_value**2)


# Figure plot
X = np.linspace(1e-6, 1e-2, 10)
Y = (10**intercept) * (X**slope)

plt.text(1e-3, 6e-4, r"$\propto \frac{dt}{T_0}$", color="dimgray", fontsize=20)

plt.plot(dt_range / v.T0, errors, color="orange")  # Results
plt.plot(X, Y, "-", color="dimgray", alpha=0.5, linewidth=4)  # Linear regression
plt.loglog()
plt.xlabel(r"$dt/T_0$")
plt.ylabel(r"$E_{\delta t}$")
plt.savefig(figure_path + "erreur", dpi=figure_dpi, format=figure_format)
plt.clf()
