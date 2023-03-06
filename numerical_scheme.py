import numpy as np
import values

def simplectic_scheme(theta0,thetadot0,omega0,dt,n):
    theta= np.zeros(n+1)
    thetadot = np.zeros(n+1)
    theta[0] = theta0
    thetadot[0] = thetadot0

    for i in range(n-1):
        thetadot[i+1] = thetadot[i] - omega0**2 * theta[i] * dt
        theta[i+1] = theta[i] + thetadot[i+1]*dt

    return theta, thetadot
