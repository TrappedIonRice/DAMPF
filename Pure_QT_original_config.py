'''
This is the 2 sites - 2 modes configuration file for the Pure QT (Pure State Quantum Trajectory) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------

Ntraj = 1000               # number of trajectories to average over
nsites = 2                 # number of sites
nosc = 2                   # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 20            # maximal bond dimension of MPS
timestep = 0.1             # integration time-step
time = 200                 # total simulation time

# -----------------------------------------------
# Initial states
# -----------------------------------------------

el_initial_state = np.array([1, 0], dtype=complex)  # initial electronic state, in the site basis

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Site energies (E_n)
energies = np.array([0.5, -0.5])

# Exchange coupling (which is the off-diagonal part of the El_Hamiltonian)
# We assume uniform coupling between all sites here for simplicity, but this can be readily changed into cases with different couplings among sites.
exchange_per_site = 0.05

# Parameters for oscillators
freqs = np.array([1, 1.2])
temps = np.array([0, 0]) # temperature are given in term of nbar, which should be much less than localDim
coups = np.array([[0.5, 0.6], [-0.5, -0.6]])
damps = np.array([0.05, 0.05])

elham = np.array([
    [energies[i] if i == j else exchange_per_site for i in range(nsites)]
    for j in range(nsites)
], dtype=float)    # exchange matrix


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
    print("elham =", elham)
    print("freqs =", freqs)
    print("temps =", temps)
    print("coups =", coups)
    print("damps =", damps)