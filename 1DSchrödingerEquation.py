# One-dimensional Schrödinger Equation
# By Sietse Buijsman

# This simulation solves the one-dimensional Schrödinger equation by means
# of a finite element method.

# Units used are:
# Distance in meters
# Time in seconds
# Other parameters are expressed in these units
# The time in the plots is expressed in minutes for brevity, but is derived
# from the unit of time, which is seconds.

#---------------------------------------------------------- Required Libraries
import matplotlib.pyplot as plt             # for plotting
import math                                 # for pi etc.
import numpy as np                          # for numpy
import numpy.linalg as la                   # for linear algebra
import scipy.sparse as sp                   # for sparse matrices

import time                                 # for a timer
import tracemalloc                          # for measuring memory usage

#--------------------------------------------------- Run Time and Memory Usage
start_time = time.time()
tracemalloc.start()

#------------------------------------------------------------------ Parameters
L = 1                                       # [m] Width of box.
T = 100                                     # [s] Total time.

Nx = 1000                                    # [] Number of grid points along Box.

hbar = 1#1.05457148e-34                       # [Js] Reduced Planck's constant
me = 1#9.10938188e-31                         # [m] Electron mass.

Alpha = hbar/(2*me)                         # [m²/s] hbar/2me.

dx = L/Nx                                   # [m] Distance step size.
ft = 100                                   # [] A dimensionless factor determining the timestep dt. Must be greater than or equal to 1 to ensure numerical stability.
dt = dx/ft#dx*dx /(ft * 2 * Alpha)                # [s] Time step size.

Nt = math.ceil(T/dt)                        # [] Number of time steps.

D = complex(0, Alpha*dt/(dx*dx))            # [] Recurring constant.
#----------------------------------------------------------- Initiate Matrices
Psi = np.array([0]*(Nx+1), dtype=complex)                         # [] Matrix representing temperature. Rows are at constant x, columns at constant t.
Hmat = sp.csr_matrix(((Nx+1), (Nx+1)), dtype = complex).toarray() # [] Time evolution matrix.
X = np.arange(0, L+dx, dx)                                        # [m] Array with all position coordinates.

#------------------------------------------------------- Initial Distributions
# Normalised gaussian distribution with:
# Centre the centre of the gaussian distribution
# Delta the width of the gaussian distribution
# x the position at which the state is to be determined
def gaussian(Wavevector, Centre, Delta, x):
    return pow(2*math.pi*Delta*Delta, -1/4) * np.exp(complex(0, Wavevector*(x - Centre))) * np.exp(-pow((x - Centre)/(2*Delta), 2))

#---------------------------------------------------------- Initial Conditions
# Using parameters:
# Centre = L/3
# Width = L/7.5
# Wavevector = 

Hmat[0, 0] = 1 - 2*D
Hmat[0, 1] = D
Hmat[Nx, Nx] = 1 - 2*D
Hmat[Nx, Nx-1] = D

for kk in range(0, Nx+1):                   # The initial condition is a Gaussian state, modulated by a wave vector.
    Psi[kk] = gaussian(1, 20, 4, kk*dx)

#----------------------------------------------------------- Fill the H-matrix
for ii in range(1, Nx):                     # The H-matrix is defined according to the finite-element method.
    Hmat[ii, ii] = 1-2*D
    Hmat[ii, ii+1] = D
    Hmat[ii, ii-1] = D

# Check for unitarity
print(Hmat * np.transpose(Hmat.conj()) - np.eye(Nx+1))

#-------------------------------------------------------------- Time Evolution
# Four times at which the probability distribution is evaluated
T2 = math.floor(Nt/5)                       # [] Time 2
T3 = math.floor(3*Nt/5)                     # [] Time 3
T4 = Nt                                     # [] Time 4

ax = plt.subplot(111)               # Initialises legend and plots initial probability distribution
Psi21 = np.real(Psi * Psi.conj())
ax.plot(X, Psi21, label='t = '+str(0)+' min')

print(Psi21.sum())

for jj in range(1, Nt+1):
    Psi = Hmat @ Psi       # The next state is calculated by multiplying the Hmatrix by the previous state.

    if jj == T2:
        Psi22 = np.real(Psi * Psi.conj())
        ax.plot(X, Psi22, label='t = '+str(math.floor(T2/60*T))+' min')

    if jj == T3:
        Psi23 = np.real(Psi * Psi.conj())
        ax.plot(X, Psi23, label='t = '+str(math.floor(T3/60*T))+' min')

    if jj == T4:
        Psi24 = np.real(Psi * Psi.conj())
        ax.plot(X, Psi24, label='t = '+str(math.floor(T4/60*T))+' min')

#-------------------------------------------------------------------- Plotting
                                    # Display run time and memory usage
print("--- %s seconds ---" % (time.time() - start_time))
peak = tracemalloc.get_traced_memory()
print(peak)
tracemalloc.stop()

                                    # Axis and graph titles
plt.title('One-Dimensional Schrödinger Equation')
plt.xlabel('Position Along Box [m]')
plt.ylabel('Probability []')

                                    # Display a grid and show the plots and legend
ax.legend()
plt.grid(True)
plt.show()