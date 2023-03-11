import numpy as np

import values


def simplectic_scheme(theta0, theta_dot0, omega0, dt, N):
    """
    Compute the value for theta and theta_dot for N iteration of duratation dt
    with an simplectic temporal scheme
    """

    theta = np.zeros(N + 1)
    theta_dot = np.zeros(N + 1)
    theta[0] = theta0
    theta_dot[0] = theta_dot0

    for i in range(N - 1):
        theta_dot[i + 1] = theta_dot[i] - omega0**2 * theta[i] * dt
        theta[i + 1] = theta[i] + theta_dot[i + 1] * dt

    return theta, theta_dot


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


def implicit_scheme(theta0, theta_dot0, omega0, dt, N):
    """
    Compute the value for theta and theta_dot for N iteration of duratation dt
    with an implicit temporal scheme
    """
    theta = theta0 * np.ones(N + 1)
    theta_dot = theta_dot0 * np.ones(N + 1)

    for n in range(N):
        theta_dot[n + 1] = (theta_dot(n) - omega0**2 * theta[n] * dt) / (
            1 + (omega0 * dt) ** 2
        )
        theta[n + 1] = (theta[n] + theta_dot[n] * dt) / (1 + (omega0 * dt) ** 2)

    return (theta, theta_dot)
