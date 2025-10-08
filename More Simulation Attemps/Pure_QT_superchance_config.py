'''
This is the 10 sites - 10 modes configuration file for the Pure QT (Pure State Quantum Trajectory) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------
Ntraj = 1000               # number of trajectories to average over
nsites = 10                # number of sites
nosc = 10                  # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 10            # maximal bond dimension of MPS
timestep = 0.1             # integration time-step
time = 200                 # total simulation time

# -----------------------------------------------
# Initial states
# -----------------------------------------------

el_initial_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)  # initial electronic state, in the site basis

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Site energies (E_n)
energies = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0])

# Exchange coupling (which is the off-diagonal part of the El_Hamiltonian)
# We assume uniform coupling between all sites here for simplicity, but this can be readily changed into cases with different couplings among sites.
exchange_per_site = 0.05

# Parameters for oscillators
freqs = np.array([1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
temps = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])# temperature are given in term of nbar, which should be much less than localDim
coups = np.array([[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
                  [-0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4],
                  [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2],
                  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
                  [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95],
                  [0.02, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92],
                  [-0.02, -0.12, -0.22, -0.32, -0.42, -0.52, -0.62, -0.72, -0.82, -0.92],
                  [0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91]])
damps = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03])

elham = np.array([[energies[i] if i == j else exchange_per_site for i in range(nsites)] for j in range(nsites)])
# elham = np.diag(energies) + np.diag([exchange_per_site]*(nsites-1), k=1) + np.diag([exchange_per_site]*(nsites-1), k=-1)    # exchange matrix

# -----------------------------------------------
# Additional jump operators and outputs
# -----------------------------------------------

N_operator = np.diag(np.arange(localDim))  # number operator for oscillators

additional_osc_jump_op_dic = {
    "1": np.sqrt(0.005) * N_operator,
    "2": np.sqrt(0.005) * N_operator,
}
additional_osc_output_dic = {
    "1": N_operator,
    "2": N_operator,
}
# The index represents the oscillator index, which starts from 1

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
    print("el_initial_state =\n", el_initial_state)
    print()
    print("elham =\n", elham)
    print("freqs =", freqs)
    print("temps =", temps)
    print("coups =\n", coups)
    print("damps =", damps)
    print()
    print("additional_osc_jump_op_dic =\n", additional_osc_jump_op_dic)
    print("additional_osc_output_dic =\n", additional_osc_output_dic)