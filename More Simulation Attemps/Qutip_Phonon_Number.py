'''
This script runs quantum trajectory simulations using QuTiP to benchmark against the Phonon number results from the Tensor Network (TN) based Pure QT method.
'''

import matplotlib.pyplot as plt
import numpy as np
from qutip import (Qobj, tensor, basis, destroy, qeye, mcsolve)
import time as timer

import Pure_QT_config as config

def run_qutip_simulation(localDim):

    print("--- Simulation Started (Using Latest User-Provided Config) ---")
    start_time = timer.time()

    dim_sys = config.nsites
    N_osc = localDim
    w0 = config.freqs[0]
    gamma = config.damps[0]
    coupling_constants = config.coups.flatten() 
    
    H_el = Qobj(config.elham, dims=[[dim_sys], [dim_sys]])

    a = destroy(N_osc)
    n_op = a.dag() * a
    ident_osc = qeye(N_osc)
    ident_sys = qeye(dim_sys)

    H_el_full = tensor(H_el, ident_osc)
    H_osc_full = tensor(ident_sys, w0 * n_op)
    
    H_int_full = Qobj(np.zeros((dim_sys * N_osc, dim_sys * N_osc)), 
                      dims=[[dim_sys, N_osc], [dim_sys, N_osc]])

    for i, g_i in enumerate(coupling_constants):
        proj_i = basis(dim_sys, i) * basis(dim_sys, i).dag()
        term_i = g_i * tensor(proj_i, a + a.dag())
        H_int_full += term_i
    
    H = H_el_full + H_osc_full + H_int_full

    psi_sys_0 = Qobj(config.el_initial_state, dims=[[dim_sys], [1]])
    psi_osc_0 = basis(N_osc, 0)
    psi0 = tensor(psi_sys_0, psi_osc_0)

    n_th = config.temps[0]
    print(f"Directly using thermal phonon number n_th = {n_th} from config file.")

    c_op1 = np.sqrt(gamma * (1 + n_th)) * a
    c_op2 = np.sqrt(gamma * n_th) * a.dag()
    c_ops_full = [tensor(ident_sys, c_op1), tensor(ident_sys, c_op2)]

    e_ops = [tensor(ident_sys, n_op)]

    tlist = np.arange(0, config.time, config.timestep)
    
    print(f"Running mcsolve for {config.Ntraj} trajectories...")
    result = mcsolve(H, psi0, tlist, c_ops=c_ops_full, e_ops=e_ops, 
                     ntraj=config.Ntraj, progress_bar=True)
    
    simulation_time = timer.time() - start_time
    print(f"--- Simulation Finished in {simulation_time:.2f} seconds ---")

    plt.plot(tlist, result.expect[0], label="localDim = {}".format(localDim))

if __name__ == "__main__":
    
    localDim_array = [5, 10, 15]
    for localDim in localDim_array:
        print(f"\n=== Running simulation with localDim = {localDim} ===")
        run_qutip_simulation(localDim)
    print("\nAll simulations completed.")
    
    plt.title("Average Phonon Number vs. Time (Latest Config)")
    plt.xlabel("Time")
    plt.ylabel("Average Phonon Number <n>")
    plt.grid(True)
    plt.legend()
    plt.savefig("phonon_number_latest_config.pdf")
    plt.show()