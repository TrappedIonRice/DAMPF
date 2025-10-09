'''
This is to benchmarking results from Diego's paper (Fig 5d).
This is the 10 sites - 2 modes default configuration file for the Pure QT (Pure State Quantum Trajectory) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------

Ntraj = 500                # number of trajectories to average over
nsites = 10                # number of sites
nosc = 1                   # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 10            # maximal bond dimension of MPS
timestep = 0.5             # integration time-step
time = 40*3.1415926*2     # total simulation time

# -----------------------------------------------
# Initial states
# -----------------------------------------------

el_initial_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)  # initial electronic state, in the site basis

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Site energies (E_n)
energies = 3 * np.array([0.5, 0.5, -0.5, -0.5, -1.5, -1.5, -2.5, -2.5, -3.5, -3.5])

# Exchange coupling (which is the off-diagonal part of the El_Hamiltonian)
# We assume uniform coupling between all sites here for simplicity, but this can be readily changed into cases with different couplings among sites.
J = 0.3

# Parameters for oscillators
freqs = np.array([1])
temps = np.array([0.01]) # temperature are given in term of nbar, which should be much less than localDim
coups = np.array([[0.5], [0.5], [-0.5], [-0.5], [0.5], [0.5], [-0.5], [-0.5], [0.5], [0.5]])
damps = np.array([0.039522])

additional_dist = np.array(
    [[0, 0, 2, 2, 4, 4, 6, 6, 8, 8],
     [0, 0, 2, 2, 4, 4, 6, 6, 8, 8],
     [2, 2, 0, 0, 2, 2, 4, 4, 6, 6],
     [2, 2, 0, 0, 2, 2, 4, 4, 6, 6],
     [4, 4, 2, 2, 0, 0, 2, 2, 4, 4],
     [4, 4, 2, 2, 0, 0, 2, 2, 4, 4],
     [6, 6, 4, 4, 2, 2, 0, 0, 2, 2],
     [6, 6, 4, 4, 2, 2, 0, 0, 2, 2],
     [8, 8, 6, 6, 4, 4, 2, 2, 0, 0],
     [8, 8, 6, 6, 4, 4, 2, 2, 0, 0]]
)

elham = np.diag(energies)
for i in range(nsites):
    for j in range(nsites):
        if i < j:
            elham[i][j] = J / (j - i + additional_dist[i][j])
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