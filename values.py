from math import pi, sqrt


# ------------------------------------------------------------
#   Choosen values
# ------------------------------------------------------------
m = 1  # Pendulum mass [kg]
l = 0.1  # Pendulum length [m]
theta0 = pi / 10  # Initial angle between the pendulum and the vertical axis [rad]
theta_dot0 = 0  # Initial angular speed of the pendulum [rad]


# ------------------------------------------------------------
#   Physical constants
# ------------------------------------------------------------
g = 9.81  # Earth gravital acceleration [m/s^2]


# ------------------------------------------------------------
#   Computed Values
# ------------------------------------------------------------
omega0 = sqrt(g / l)  # Pendulum angular frequency [rad/s]
T0 = 2 * pi / omega0  # Pendulum period [s]
