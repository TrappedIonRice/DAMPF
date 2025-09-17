'''
This is the 2 sites - 2 modes configuration file for the Rho Fixed Step (Density Matrix Fixed Step) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------
nsites = 2                 # number of sites
nosc = 2                   # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 10            # maximal bond dimension of MPS
timestep = 0.2             # integration time-step
time = 200                 # total simulation time

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Site energies (E_n)
energies = [0.5, -0.5]

# Exchange coupling (which is the off-diagonal part of the El_Hamiltonian)
# We assume uniform coupling between all sites here for simplicity, but this can be readily changed into cases with different couplings among sites.
exchange_per_site = 0.05
elham = np.diag(energies) + exchange_per_site * (np.ones((nsites, nsites)) - np.eye(nsites))

# Parameters for oscillators
freqs = np.array([1, 1.2])
temps = np.array([0, 0]) # temperature are given in term of nbar, which should be much less than localDim
coups = np.array([[0.5, 0.6], [-0.5, -0.6]])
damps = np.array([0.05, 0.05])

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
    print("elham =\n", elham)
    print("freqs =", freqs)
    print("temps =", temps)
    print("coups =", coups)
    print("damps =", damps)