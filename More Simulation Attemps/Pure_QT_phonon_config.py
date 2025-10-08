'''
This is the 2 sites - 2 modes configuration file for the Pure QT (Pure State Quantum Trajectory) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------

Ntraj = 1000                # number of trajectories to average over
nsites = 4                 # number of sites
nosc = 1                   # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 10            # maximal bond dimension of MPS
timestep = 0.1             # integration time-step
time = 50*3.1415926*2      # total simulation time

# -----------------------------------------------
# Initial states
# -----------------------------------------------

el_initial_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=complex)  # initial electronic state, in the site basis

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Site energies (E_n)
energies = 3 * np.array([0.5, 0.5, -0.5, -0.5])

# Exchange coupling (which is the off-diagonal part of the El_Hamiltonian)
# We assume uniform coupling between all sites here for simplicity, but this can be readily changed into cases with different couplings among sites.
J = 0.3

# Parameters for oscillators
freqs = np.array([1])
temps = np.array([0.01]) # temperature are given in term of nbar, which should be much less than localDim
coups = np.array([[0.5], [0.5], [-0.5], [-0.5]])*2
damps = np.array([0.039552])

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
# Additional jump operators and outputs
# -----------------------------------------------

# additional_osc_jump_op_dic = {
#     "1": np.sqrt(0.005) * N_operator,
#     "2": np.sqrt(0.005) * N_operator,
# }
additional_osc_jump_op_dic = {}
additional_osc_output_dic = {}
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