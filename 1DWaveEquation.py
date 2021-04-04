# One-dimensional Heat Equation
# By Sietse Buijsman

# This simulation solves the one-dimensional heat equation by means of a
# finite element method.

# Units used are:
# Distance in meters
# Time in seconds
# Temperature in Kelvin
# Other parameters are expressed in these units
# The time in the plots is expressed in minutes for brevity, but is derived
# from the unit of time, which is seconds.

#---------------------------------------------------------- Required Libraries
import matplotlib.pyplot as plt             # for plotting
import math                                 # for pi etc.
import numpy as np                          # for numpy
import scipy.sparse as sp                   # for sparse matrices
from scipy import linalg as la              # for linear algebra

import time                                 # for a timer
import tracemalloc                          # for measuring memory usage

#--------------------------------------------------- Run Time and Memory Usage
start_time = time.time()
tracemalloc.start()

#------------------------------------------------------------------ Parameters
L = 1.5                                     # [m] length of string
frequency = 440                             # [Hz] frequency of wave
wavelength = 2*L                            # [m] wavelength of wave
omega = 2*math.pi*frequency                 # [rad/s] angular frequency of wave
c = wavelength*frequency                    # [m/s] speed of wave

                                            # reports the speed of the wave
print('The speed of the wave, c = %s m/s' % c)

#------------------------------------------------------- Simulation Parameters
Nx = 1000                                   # [] number of grid points
Nt = 500                                    # [] number of time steps

dx = L/Nx                                   # [m] distance step
dt = .7*dx/c                                # [s] time step

T = Nt*dt                                   # [s] total simulation time

alpha = pow(dt*c/dx, 2)                     # [] recurring constant

print('alpha = %s' % alpha)
print('dt = %s s' % dt)
print('dx/c = %s s' % (dx/c))

#--------------------------------------------------------- Initialise Matrices
                                            # [] time evolution matrix
                                            #sp.csr_matrix(((Nx+1), (Nx+1)), dtype = float).toarray()
Hmat = np.array([[0.]*(Nx+1)]*(Nx+1))
uxt = np.array([0.]*(Nx+1))                 # [m] displacement array at t
wxt = np.array([0.]*(Nx+1))                 # [m] displacement array at t-dt
vxt = np.array([0.]*(Nx+1))                 # [m/s] derivative of displacement array
X = np.arange(0, L + dx, dx)                # [m] position array

#---------------------------------------------------------- Initial Conditions
# Fill H-matrix
Hmat[0, 0] = 2 - 2*alpha
Hmat[0, 1] = alpha

Hmat[Nx, Nx] = 2 - 2*alpha
Hmat[Nx, Nx-1] = alpha

for ii in range(1, Nx-1):
    Hmat[ii, ii] = 2 - 2*alpha
    Hmat[ii, ii-1] = alpha
    Hmat[ii, ii+1] = alpha

# Initial Displacement
for jj in range(0, Nx+1):                   # initial condition is a sine wave
    uxt[jj] = 0.0001 * math.sin(math.pi*jj*dx/L)

# Initial Displacement Derivative
for kk in range(0, Nx+1):                   # initial condition is stationary
    vxt[kk] = 0

# Displacement after t = dt
wxt = uxt
uxt = 1/2*(Hmat @ uxt - uxt) + vxt*dt                   # second displacement
vxt = uxt

#-------------------------------------------------------------- Time Evolution
# Four times at which the displacement is evaluated
T2 = Nt/5                                   # [] Time 2
T3 = 3*Nt/5                                 # [] Time 3
T4 = Nt-1                                   # [] Time 4

# Initialises legend and plots the initial state
ax = plt.subplot(111)
ax.plot(X, uxt, label='t = '+str('0 s'))

for ll in range(2, Nt):
    if ll%2 == 0:
        uxt = Hmat @ uxt - vxt
        vxt = uxt

    else:
        uxt = Hmat @ uxt - wxt
        wxt = uxt

        # At T2, T3 and T4, uxt is plotted.
    if ll == T2:
        ax.plot(X, uxt, label='t = '+str(T*T2/(60*Nt))+' s')

    if ll == T3:
        ax.plot(X, uxt, label='t = '+str(T*T3/(60*Nt))+' s')

    if ll == T4:
        ax.plot(X, uxt, label='t = '+str(T*T4/(60*Nt))+' s')

#-------------------------------------------- Report Run Time and Memory Usage
print("--- %s seconds ---" % (time.time() - start_time))
peak = tracemalloc.get_traced_memory()
print(peak)
tracemalloc.stop()

#-------------------------------------------------------------------- Plotting
                                    # Axis and graph titles
plt.title('One-Dimensional String')
plt.xlabel('Position Along String [m]')
plt.ylabel('Displacement [m]')

                                    # Display a grid and show the plots and legend
ax.legend()
plt.grid(True)
plt.show()