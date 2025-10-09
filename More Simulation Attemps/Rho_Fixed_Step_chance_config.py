'''
This is the 5 sites - 5 modes default configuration file for the Rho Fixed Step (Density Matrix Fixed Step) DAMPF simulation.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------
nsites = 5                 # number of sites
nosc = 5                   # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 10            # maximal bond dimension of MPS
timestep = 1               # integration time-step
time = 10                  # total simulation time

# -----------------------------------------------
# Initial states
# -----------------------------------------------

el_initial_state = np.array([1, 0, 0, 0, 0], dtype=complex)  # initial electronic state, in the site basis

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
    print("timestep =", timestep)
    print("time =", time)
    print()
    print("el_initial_state =", el_initial_state)
    print()
    print("elham =\n", elham)
    print("freqs =", freqs)
    print("temps =", temps)
    print("coups =", coups)
    print("damps =", damps)