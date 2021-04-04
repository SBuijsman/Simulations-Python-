# Harmonic Oscillator by Sietse Buijsman
# Numerical analysis of the driven and damped harmonic oscillator
# by means of the Runge-Kutta 4th order algorithm.

# Required libraries ------------------------------------------------------------------------------
import matplotlib.pyplot as plt     # for plotting
import math                         # for pi etc.
import numpy as np                  # for numpy

# Parameters --------------------------------------------------------------------------------------
mass = 2                            # [kg] mass of the oscillator, m > 0
k = 3                               # [N/m] spring constant of oscillator, k > 0
d = 0.10                            # [Ns/m] damping constant of oscillator
omega2 = k/mass                     # [1/s²] eigenfrequency of oscillator squared
omega = pow(omega2, 1/2)            # [1/s] eigenfrequency of oscillator
omegad = omega                      # [1/s] driving frequency
F0 = 5                              # [N] driving amplitude
phase = math.pi/2                   # [rad] phase of driving force

# Simulation parameters ---------------------------------------------------------------------------
Nfreq = 50                          # [] number of timesteps per oscillation
Nosc = 30                           # [] number of oscillations to simulate
Nt = Nfreq * Nosc                   # [] number of time steps in total

dt = 2*math.pi / (Nfreq * omega)    # [s] timestep of simulation
dt2 = pow(dt, 2)                    # [s²] timestep squared

A0 = F0*dt2/mass                    # [m] reduced driving amplitude

# Initial Conditions ------------------------------------------------------------------------------
xi = 1                              # [m] initial displacement
vi = 0                              # [m/s] initial speed

# Arrays ------------------------------------------------------------------------------------------
Y = np.array([[0.]*Nt, [0.]*Nt])    # [m, m/s] state vector
Y[0,0] = xi                         # [m] insert initial position
Y[0,1] = vi                         # [m/s] insert initial speed
T = [jj*dt for jj in range(0, Nt)]  # [s] time list

# Functions ---------------------------------------------------------------------------------------
def f1(t, x, v):                    # speed function defined as dx/dt = v = f1(t, x, v)
    return v

def f2(t, x, v):                    # acceleration function defined as dv/dt = -omega2*x - d*v + A0*math.cos(omegad*t) = f1(t, x, v)
    return -omega2*x - d*v + A0*math.cos(omegad*t + phase)

def kvector(t, x, v):               # k-vector function, returning a vector with f1 and f2
    kx = f1(t, x, v)
    kv = f2(t, x, v)

    ktot = np.array([kx, kv])

    return ktot

# Loop --------------------------------------------------------------------------------------------
for ii in range(0, Nt-1):
    k1 = kvector(T[ii],        Y[0, ii],              Y[1, ii])
    k2 = kvector(T[ii] + dt/2, Y[0, ii] + dt*k1[0]/2, Y[1, ii] + dt*k1[1]/2)
    k3 = kvector(T[ii] + dt/2, Y[0, ii] + dt*k2[0]/2, Y[1, ii] + dt*k2[1]/2)
    k4 = kvector(T[ii] + dt,   Y[0, ii] + dt*k3[0],   Y[1, ii] + dt*k3[1])

    Y[:, ii + 1] = Y[:, ii] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Plotting ----------------------------------------------------------------------------------------
ax = plt.subplot(111)                       # for the legend
ax.plot(T, Y[0, :], label='Displacement')   # plot position against time
ax.plot(T, Y[1, :], label='Speed')          # plot speed against time

plt.title('Harmonic Damped and Driven Oscillator')
plt.xlabel('Time [s]')
plt.ylabel('Deflection [m]')

ax.legend()

plt.grid(True)
plt.show()                                  # display plot