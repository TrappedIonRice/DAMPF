'''
This is the 5 sites - 5 modes configuration file for the Pure QT (Pure State Quantum Trajectory) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------
Ntraj = 100                # number of trajectories to average over
nsites = 5                 # number of sites
nosc = 5                   # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 20            # maximal bond dimension of MPS
timestep = 0.1             # integration time-step
time = 200                 # total simulation time

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Site energies (E_n)
energies = np.array([2.0, 1.0, 0.5, -0.5, -1.0])

# Exchange coupling (which is the off-diagonal part of the El_Hamiltonian)
# We assume uniform coupling between all sites here for simplicity, but this can be readily changed into cases with different couplings among sites.
exchange_per_site = 0.05

# Parameters for oscillators
freqs = np.array([1, 1.2, 1.4, 1.6, 1.8])
temps = np.array([0, 0, 0, 0, 0]) # temperature are given in term of nbar, which should be much less than localDim
coups = np.array([[0.5, 0.6, 0.7, 0.8, 0.9], 
                [-0.5, -0.6, -0.7, -0.8, -0.9], 
                [-0.3, -0.4, -0.5, -0.6, -0.7], 
                [0.3, 0.4, 0.5, 0.6, 0.7], 
                [0.1, 0.2, 0.3, 0.4, 0.5]])
damps = np.array([0.05, 0.05, 0.05, 0.05, 0.05])

elham = np.diag(energies) + np.diag([exchange_per_site]*(nsites-1), k=1) + np.diag([exchange_per_site]*(nsites-1), k=-1)    # exchange matrix

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
    print("elham =", elham)
    print("freqs =", freqs)
    print("temps =", temps)
    print("coups =", coups)
    print("damps =", damps)