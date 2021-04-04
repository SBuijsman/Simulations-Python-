# Harmonic Oscillator by Sietse Buijsman
# Numerical analysis of the driven and damped harmonic oscillator
# by means of the Euler method.

# Required libraries
import matplotlib.pyplot as plt     # for plotting
import math                         # for pi etc.

# Parameters
mass = 2                            # [kg] mass of the oscillator
k = 3                               # [N/m] spring constant of oscillator
d = 0.1                             # [Ns/m] damping constant of oscillator
omega2 = k/mass                     # [1/s²] eigenfrequency of oscillator squared
omega = pow(omega2, 1/2)            # [1/s] eigenfrequency of oscillator
omegad = .8*omega                   # [1/s] driving frequency
F0 = 2                              # [N] driving amplitude

# Simulation parameters
Nfreq = 50                          # [] Number of timesteps per oscillation
Nosc = 30                           # [] Number of oscillations to simulate
Nt = Nfreq * Nosc                   # [] Number of time steps in total

dt = 2*math.pi / (Nfreq * omega)    # [s] timestep of simulation
dt2 = pow(dt, 2)                    # [s²] timestep squared

A0 = F0*dt2/mass                    # [m] reduced driving amplitude

# Initial Conditions
xi = 1                              # [m] initial displacement
vi = 2                              # [m/s] initial speed

# Arrays
X = []                              # [m] position list
T = [jj*dt for jj in range(0, Nt)]  # [s] time list

X.append(xi)                        # set first value of X to be the initial position
X.append(X[0] + vi*dt)              # set second value of X to be the initial speed 

# Loop
for ii in range(1, Nt-1):
    xnew = ((2 - dt2*omega2 - d*dt)*X[ii] - X[ii - 1] + A0*math.cos(omegad*T[ii])) / (1 + d*dt)
    X.append(xnew)

plt.plot(T, X)                      # plot position against time
ax = plt.subplot(111)               # for the legend
ax.plot(T, X, label='Harmonic Oscillator')
plt.title('Harmonic Damped and Driven Oscillator')
plt.xlabel('Time [s]')
plt.ylabel('Deflection [m]')
ax.legend()
plt.grid(True)
plt.show()                          # display plot