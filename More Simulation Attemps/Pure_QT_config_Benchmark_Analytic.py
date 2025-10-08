'''
This is the 2 sites - 7 modes configuration file for the Pure QT (Pure State Quantum Trajectory) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np
import math


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------

Normalization_Factor = 100
Ntraj = 1000               # number of trajectories to average over
nsites = 2                 # number of sites
nosc = 7                   # total number of oscillators
localDim = 10              # local dimension of oscillators
maxBondDim = 10            # maximal bond dimension of MPS
timestep = 0.00001 * Normalization_Factor           # integration time-step
time = 1 * Normalization_Factor                    # total simulation time

# -----------------------------------------------
# Initial states
# -----------------------------------------------

el_initial_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)  # initial electronic state, in the site basis

# -----------------------------------------------
# Parameters for system dynamics
# -----------------------------------------------

# Parameters for oscillators
freqs = np.array([107.96573614, 151.35253695, 75.31574205, 29.84056346, 49.98315227, 208.41392941, 278.54598055]) / Normalization_Factor
temps = np.array([5] * nosc)# temperature are given in term of nbar, which should be much less than localDim
coups = np.array([[23.70902183, 28.04179863, 19.16124301, 9.58555597, 14.50739058, 26.61600313, 22.55370035], [-23.70902183, -28.04179863, -19.16124301, -9.58555597, -14.50739058, -26.61600313, -22.55370035]]) / Normalization_Factor
damps = np.array([30.81591027, 42.58938392, 22.08333054, 10.28095595, 15.55502598, 50.0, 50.0]) / Normalization_Factor


elham = np.zeros((nsites, nsites), dtype=complex)

# -----------------------------------------------
# Additional jump operators and outputs
# -----------------------------------------------

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