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
import numpy.linalg as la                   # for linear algebra
import scipy.sparse as sp                   # for sparse matrices

import time                                 # for a timer
import tracemalloc                          # for measuring memory usage

#--------------------------------------------------- Run Time and Memory Usage
start_time = time.time()
tracemalloc.start()

#------------------------------------ Physical Constants & Material parameters
L = 5                                       # [m] Length of rod.
R = 0.02                                    # [m] Radius of rod.

rho = 8960                                  # [kg/m³] Density of copper.
Alpha = 11e-6                               # [m²/s] Thermal diffusivity of copper.
C_cu = 380                                  # [J/kgK] Specific heat capacity of copper.

mass = rho * math.pi*R*R*L                  # [kg] Mass of rod.
C_tot = C_cu * mass                         # [J/K] Heat capacity of rod.

#------------------------------------------------------------------ Parameters
T = 100000                                  # [s] Total time.

Nx = 300                                    # [] Number of grid points along rod.

dx = L/Nx                                   # [m] Distance step size.
ft = 2                                      # [] A dimensionless factor determining the timestep dt. Must be greater than or equal to 1 to ensure numerical stability.
dt = dx*dx /(ft * 2 * Alpha)                # [s] Time step size.

print(dt)

Nt = math.ceil(T/dt)                        # [] Number of time steps.

D = Alpha*dt/(dx*dx)                        # [] Recurring constant.

H = 50                                      # [W] Heat added per second.

#----------------------------------------------------------- Initiate Matrices
uxt = np.array([0.]*(Nx+1))                                     # [K] Matrix representing temperature.
Hmat = sp.csr_matrix(((Nx+1), (Nx+1)), dtype = float).toarray() # [] Time evolution matrix.
Heat = np.array([0.]*(Nx+1))                                    # [K/s] Heat array.
X = np.arange(0, L+dx, dx)                                      # [m] Array with all position coordinates.

#------------------------------------------------------- Initial Distributions
# Gaussian distribution with:
# T0 the base temperature
# Tmax the height of the gaussian peak
# Centre the centre of the gaussian distribution
# Delta the width of the gaussian distribution
# x the position at which the temperature is to be determined
def gaussian(T0, Tmax, Centre, Delta, x):
    return T0 + Tmax * math.exp(-pow((x - Centre)/Delta, 2))

# Sinusoidal distribution with:
# T0 the base temperature
# Tmax the amplitude of the sine
# Length the length of the rod
# x the position at which the temperature is to be determined
def sine(T0, Tmax, Length, x):
    return T0 + Tmax * math.sin(2*math.pi * x/Length)

# Linear distribution with:
# T0 the base temperature
# Slope the slope of the linear distribution
# x the position at which the temperature is to be determined
def linear(T0, Slope, x):
    return T0 + Slope*x

#---------------------------------------------------------- Initial Conditions
# Using parameters:
# T0 = 300 K
# Tmax = 200 K
# Centre = L/3
# Width = L/7.5
# Slope = 40 K/m

Hmat[0, 1] = 1                              # This boundary condition ensures that du/dx(x=0) = 0 at every timestep.
Hmat[Nx, Nx-1] = 1                          # This boundary condition ensures that du/dx(x=L) = 0 at every timestep.

for kk in range(0, Nx+1):                   # The initial condition is a Gaussian temperature distribution.
    uxt[kk] = gaussian(300, 200, L/3, L/7.5, kk*dx)          # Gaussian
    #uxt[0, kk] = sine(300, 200, L, kk*dx)                    # Sine
    #uxt[0, kk] = linear(300, 40, kk*dx)                      # Linear

#----------------------------------------------------------- Fill the H-matrix
for ii in range(1, Nx):                     # The H-matrix is defined according to the finite-element method.
    Hmat[ii, ii] = 1-2*D
    Hmat[ii, ii+1] = D
    Hmat[ii, ii-1] = D

#------------------------------------------------------- Define the heat array
for mm in range(math.floor(Nx/3), math.floor(Nx/3) + 10):
    Heat[mm] = H/C_tot*dt

#-------------------------------------------------------------- Time Evolution
# Four times at which the temperature distribution is evaluated
T2 = math.floor(Nt/5)                       # [] Time 2
T3 = math.floor(3*Nt/5)                     # [] Time 3
T4 = Nt                                     # [] Time 4

# Initialises legend and plots the initial state
ax = plt.subplot(111)
ax.plot(X, uxt, label='t = '+str('0 min'))

for jj in range(1, Nt+1):
    uxt = Hmat @ uxt + Heat                  # The next state is calculated by multiplying the Hmatrix by the previous state.

    # At T2, T3 and T4, uxt is plotted.
    if jj == T2:
        ax.plot(X, uxt, label='t = '+str(math.floor(T*T2/(60*Nt)))+' min')

    if jj == T3:
        ax.plot(X, uxt, label='t = '+str(math.floor(T*T3/(60*Nt)))+' min')

    if jj == T4:
        ax.plot(X, uxt, label='t = '+str(math.floor(T*T4/(60*Nt)))+' min')
#-------------------------------------------------------------------- Plotting
                                    # Axis and graph titles
plt.title('One-Dimensional Heat Conductor')
plt.xlabel('Position Along Rod [m]')
plt.ylabel('Temperature [K]')

                                    # Display run time and memory usage
print("--- %s seconds ---" % (time.time() - start_time))
peak = tracemalloc.get_traced_memory()
print(peak)
tracemalloc.stop()

                                    # Display a grid and show the plots and legend
ax.legend()
plt.grid(True)
plt.show()