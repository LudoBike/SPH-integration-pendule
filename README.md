# SPH-integration-pendule
Simulation of a simple pendulum and comparaision of severals temporal numerical schemes.

The considered schemes are :
 - Explicit scheme
 - Implicit scheme
 - Simplectic scheme

# Dependancies
 - numpy
 - pyplot
 - scipy.stats

# Résults
For a pendulum of mass $m$, length $l$, in a vertical gravitationnal field $g$. We denote $\omega_0$ its angular frequency and $T_0$ its period.

## Phase diagram for several numerical schemes
![Phase diagram for several numerical schemes](https://user-images.githubusercontent.com/14591631/229366340-f46cf3de-398a-4109-839c-1c1fbaf9e592.png)

## Time evolution of the hamiltonien of the pendulum
![Time evolution of the hamiltonien of the pendulum](https://user-images.githubusercontent.com/14591631/229366348-20470a63-47f2-486f-88f0-fe0f9c8cdd17.png)

## Error in L2 norm as a function of the timestep dt for the simplectic scheme
![Error in L2 norm as a function of the timestep dt for the simplectic scheme](https://user-images.githubusercontent.com/14591631/229366358-8e6b97df-9bc9-4152-9921-1cdfe7422bb2.png)
