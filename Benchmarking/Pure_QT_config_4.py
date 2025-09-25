'''
This is the 2 sites - 2 modes configuration file for the Pure QT (Pure State Quantum Trajectory) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------

Ntraj = 500               # number of trajectories to average over
nsites = 4                 # number of sites
nosc = 1                   # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 10            # maximal bond dimension of MPS
timestep = 0.5             # integration time-step
time = 200*3.1415926*2     # total simulation time

# -----------------------------------------------
# Initial states
# -----------------------------------------------

el_initial_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=complex)  # initial electronic state, in the site basis

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Site energies (E_n)
energies = np.array([0.5, 0.5, -0.5, -0.5])

# Exchange coupling (which is the off-diagonal part of the El_Hamiltonian)
# We assume uniform coupling between all sites here for simplicity, but this can be readily changed into cases with different couplings among sites.
J = 0.03

# Parameters for oscillators
freqs = np.array([1])
temps = np.array([0.01]) # temperature are given in term of nbar, which should be much less than localDim
coups = np.array([[0.5], [0.5], [-0.5], [-0.5]])
damps = np.array([0.015])

elham = np.diag(energies)
for i in range(nsites):
    for j in range(nsites):
        if i < j:
            if (i <= nsites // 2 - 1) and (j >= nsites // 2):
                elham[i][j] = J / (j - i + 2)
            else:
                elham[i][j] = J / (j - i)
        elif i > j:
            elham[i][j] = elham[j][i].conjugate()
        


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