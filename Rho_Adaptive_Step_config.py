'''
This is the 2 sites - 2 modes configuration file for the Rho Adaptive Step (Density Matrix Adaptive Step) DAMPF simulation.
All of the tunable parameters are deliberately set here in a separate file.
'''


import numpy as np


# -----------------------------------------------
# Basic system parameters
# -----------------------------------------------
nsites = 2                  # number of sites
nosc = 2                    # total number of oscillators
localDim = 10               # local dimension of oscillators
maxBondDim = 10             # maximal bond dimension of MPS

# -----------------------------------------------
# Initial states
# -----------------------------------------------

el_initial_state = np.array([1, 0], dtype=complex)  # initial electronic state, in the site basis

# -----------------------------------------------
# Parameters for time evolution
# -----------------------------------------------
initial_dt = 0.5            # integration time-step
time = 200                  # total simulation time
error_tolerance = 1e-4      # error tolerance for adaptive time-stepping
S1 = 0.9                    # time-step adjustment parameter 1 (0 < S1 < 1)
S2 = 4                      # time-step adjustment parameter 2 (S2 > 1)

# We need the half-valued array of time steps for error calculation in the adaptive time-stepping scheme.
dt_array = np.linspace(0.1, 0.6, 20)
dt_array = np.vstack((dt_array, dt_array/2))

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
# Additional outputs
# -----------------------------------------------

N_operator = np.diag(np.arange(localDim))  # number operator for oscillators

additional_osc_output_dic = {
    "1": N_operator,
    "2": N_operator,
}

# -----------------------------------------------
# Print out all parameters
# -----------------------------------------------

if __name__ == "__main__":
    print("nsites =", nsites)
    print("nosc =", nosc)
    print("localDim =", localDim)
    print("maxBondDim =", maxBondDim)
    print()
    print("el_initial_state =", el_initial_state)
    print()
    print("initial_dt =", initial_dt)
    print("time =", time)
    print("error_tolerance =", error_tolerance)
    print("S1 =", S1)
    print("S2 =", S2)
    print("dt_array =", dt_array)
    print()
    print("elham =\n", elham)
    print("freqs =", freqs)
    print("temps =", temps)
    print("coups =", coups)
    print("damps =", damps)