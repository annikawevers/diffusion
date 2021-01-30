import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.animation as animation

'''
The following program models the diffusion of a drop of dye that is placed
in the center of a square dish of water. The diffustion equation is
implemented in a for loop and an animation of the evolution of dye
concentration in the dish is created. In addition, a plot of the
standard deviation of a Gaussian curve of the data vs the
square root of time is created.
'''

# constants
L = 1  # size of "box"
D = 1e-3
N = 100  # number of grid points in one-d
dt = 1e-2
dx = L/N
dy = L/N
tmax = 100
steps = int(tmax/dt)+1

# create initial conditions:
C = np.zeros((N, N))

# Set particles in a blob in the center:
C[(N//2), (N//2)] = 10
k = dt/dx/dx*D
Cp = np.zeros((N, N))
isum = np.sum(C)
xdata = np.linspace(0, L, N, endpoint=True)


def gs(x, mu, sig, a):
    '''This function takes xdata, mean, standard deviation, and amplitude
    and returns a gaussian curve. This will be used in the curve fit.'''
    return a*np.exp(-((x - mu)**2) / (2 * sig**2))


# Part 1-3
fig = plt.figure()
ims = []
new = []  # initial empty array
newtime = []
for i in range(steps):
    # inside points
    Cp[1:-1, 1:-1] = (C[1:-1, 1:-1] + k * (np.roll(C[1:-1, 1:-1], -1, axis=0)
                      + np.roll(C[1:-1, 1:-1], 1, axis=0) - 2 * C[1:-1, 1:-1])
                      + k * (np.roll(C[1:-1, 1:-1], -1, axis=1)
                      + np.roll(C[1:-1, 1:-1], 1, axis=1) - 2 * C[1:-1, 1:-1]))
    # boundaries:
    Cp[:, 0] = Cp[:, 1]
    Cp[:, -1] = Cp[:, -2]
    Cp[0, :] = Cp[1, :]
    Cp[-1, :] = Cp[-2, :]
    C, Cp = Cp, C  # swap C and Cp so they aren't the same array
    if i <= 1000 and i*dt > 0.01:
        if i % 20 == 0:
            popt, pcov = curve_fit(gs, xdata, C[:, (N//2)], p0=[(L/2), .2, 10])
            new.append(popt[1])
            newtime.append(i*dt)
    if i % 10 == 0:
        ims.append((plt.pcolormesh(C.copy()), ))
imani = animation.ArtistAnimation(fig, ims, interval=50, repeat=False)
plt.title('Evolution of C(x,y,t)')
imani.save('converges.mp4')
plt.clf()

# Plot graph
''' Threw out a few points at the beginning that did not fit surface'''
y = np.array(new)
x = np.sqrt(np.array(newtime))
z = np.sqrt(2*D)*x
plt.plot(x[2:], y[2:], 'o', label='Gaussian')
plt.plot(x, z, label='Square root(2Dt)')
axes = plt.gca()
axes.set_ylim([0, 0.15])
plt.legend()
plt.title('Diffusion')
plt.xlabel('Square Root of Time')
plt.ylabel('Standard deviation of Gaussian')
plt.savefig('diffusion.pdf')
