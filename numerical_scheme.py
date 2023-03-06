import numpy as np


def explicit_scheme(theta0, theta_dot0, omega0, dt, N):
    """
    Compute the value for theta and theta_dot for N iteration of duratation dt
    with an explicit temporal scheme
    """
    theta = theta0 * np.ones(N + 1)
    theta_dot = theta_dot0 * np.ones(N + 1)

    for n in range(N):
        theta_dot[n + 1] = theta_dot[n] - omega0**2 * theta[n] * dt
        theta[n + 1] = theta[n] + theta_dot[n] * dt

    return (theta, theta_dot)
