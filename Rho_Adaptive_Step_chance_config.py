'''
This is the 5 sites - 5 modes configuration file for the Rho Adaptive Step (Density Matrix Adaptive Step) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------
nsites = 5                 # number of sites
nosc = 5                   # total number of oscillators
localDim = 5               # local dimension of oscillators
maxBondDim = 3             # maximal bond dimension of MPS

# -----------------------------------------------
# Parameters for time evolution
# -----------------------------------------------

initial_dt = 0.5           # integration time-step
time = 20                  # total simulation time
error_tolerance = 1e-4     # error tolerance for adaptive time-stepping
S1 = 0.9
S2 = 4

# We need the half-valued array of time steps for error calculation in the adaptive time-stepping scheme.
dt_array = np.linspace(0.1, 0.6, 20)
dt_array = np.vstack((dt_array, dt_array/2))

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Site energies (E_n)
energies = [5,4.7,4.5,4.3,4]

# Exchange coupling (which is the off-diagonal part of the El_Hamiltonian)
# We assume uniform coupling between all sites here for simplicity, but this can be readily changed into cases with different couplings among sites.
exchange_per_site = 0.05

elham = np.diag(energies) + np.diag([exchange_per_site]*(nsites-1), k=1) + np.diag([exchange_per_site]*(nsites-1), k=-1)    # exchange matrix

# Parameters for oscillators
freqs = np.array([1, 1.2, 1.4, 1.6, 1.8])   # frequencies of oscillators
temps = np.array([0, 0, 0, 0, 0])  # temperature are given in term of nbar, which should be much less than localDim
coups = np.array([[1, 1.2, 1.4, 1.6, 1.8],
                  [1, 1.2, 1.4, 1.6, 1.8],
                  [1, 1.2, 1.4, 1.6, 1.8],
                  [1, 1.2, 1.4, 1.6, 1.8],
                  [1, 1.2, 1.4, 1.6, 1.8]])
damps = np.array([0.05,0.05,0.05,0.05,0.05])

# -----------------------------------------------
# Print out all parameters
# -----------------------------------------------

if __name__ == "__main__":
    print("nsites =", nsites)
    print("nosc =", nosc)
    print("localDim =", localDim)
    print("maxBondDim =", maxBondDim)
    print()
    print("initial_dt =", initial_dt)
    print("time =", time)
    print("error_tolerance =", error_tolerance)
    print("S1 =", S1)
    print("S2 =", S2)
    print("dt_array =", dt_array)
    print()
    print("elham =\n", elham)
    print("freqs =", freqs)
    print("temps =", temps)
    print("coups =", coups)
    print("damps =", damps)