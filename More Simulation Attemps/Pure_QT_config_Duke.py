'''
This is the 2 sites - 2 modes configuration file for the Pure QT (Pure State Quantum Trajectory) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------

Ntraj = 2000               # number of trajectories to average over
nsites = 2                 # number of sites
nosc = 2                   # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 10            # maximal bond dimension of MPS
timestep = 0.01            # integration time-step (timestep needs to be tuned down when the temperatures of oscillators are high)
time = 80                  # total simulation time

# -----------------------------------------------
# Initial states
# -----------------------------------------------

el_initial_state = np.array([1, 0], dtype=complex)  # initial electronic state, in the site basis

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Parameters for oscillators
freqs = None
temps = np.array([0.036, 0.036]) # temperature are given in term of nbar, which should be much less than localDim
coups = None
damps = None

DELTA = 1  # exchange coupling strength
elham = DELTA * np.array([[0, 1], [1, 0]], dtype=float) / 2   # exchange matrix

# -----------------------------------------------
# Additional jump operators and outputs
# -----------------------------------------------

additional_osc_jump_op_dic = {}
additional_osc_output_dic = {}

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