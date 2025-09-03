'''
This module contains utility functions for convenience in the simulation.
'''


import quimb.tensor as qtn
import numpy as np
import config

# Construct annihilation and creation operators for a single harmonic oscillator
def annihilation_operator(N, dtype=complex):
    
    a = np.zeros((N, N), dtype=dtype)
    # 0-based: a[n-1, n] = sqrt(n)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a

a= annihilation_operator(config.localDim)
a_dag = a.conj().T
N_operator = a_dag @ a
N1_operator = a @ a_dag

def create_thermal_mps(nosc, localDim, temps, freqs):
    
    # List to store the single-site flattened density matrices
    rho_list = []
    # bond_inds = [f'b{i}' for i in range(nosc+1)]
    
    for i in range(nosc):
        
        # Avoid division by zero for zero temperature
        if temps[i] == 0:
            rho = np.zeros((localDim, localDim))
            rho[0, 0] = 1.0
            rho_vec = rho.flatten()
            rho_list.append(rho_vec)
            continue
        
        beta = 1.0 / temps[i]  # inverse temperature
        omega = freqs[i]       # oscillator frequency
        
        # --- 1. Construct single oscillator thermal density matrix ---
        n = np.arange(localDim)
        energies = omega * (n + 0.5)               # energy levels
        boltzmann = np.exp(-beta * energies)       # Boltzmann factors  
        Z = np.sum(boltzmann)                      # partition function
        rho = np.diag(boltzmann / Z)               # thermal density matrix
        
        # --- 2. Flatten the density matrix into a vector ---
        rho_vec = rho.flatten()
        rho_list.append(rho_vec)
        
    # --- 3. Create MPS from the list of flattened density matrices ---
    thermal_mps = qtn.MPS_product_state(rho_list)
    
    # site_tensor = qtn.Tensor(rho_vec.reshape([1,localDim**2,1]), inds=(bond_inds[i],f'phy{i}',bond_inds[i+1]), tags={f's{i}'})
    
    # thermal_mps = site_tensor if i == 0 else (thermal_mps | site_tensor)
 
    return thermal_mps

# def trace_MPS(mps, nsites, localDim):
    
#     trace_vec = np.zeros(localDim**2, dtype=complex)
#     trace_vec[::localDim+1] = 1  # set diagonal elements to 1

#     trace_tensors = []
#     for i in range(nsites):
#         phys_ind = mps.site_ind(i)
#         tr_t = qtn.Tensor(trace_vec, inds=(phys_ind,))
#         trace_tensors.append(tr_t)

#     all_tensors = list(mps.tensors) + trace_tensors

#     res = qtn.tensor_contract(*all_tensors, optimize='auto-hq')
    
#     return complex(res)

# The trace of an MPS (sitewisely flattened MPO) cannot be calculated directly using the built-in function in quimb, so we define our own function here.
# We are calculating the trace by contracting the MPS with a series of identical vectors that pick out the diagonal elements.
def trace_MPS(mps, nosc, localDim):
    
    trace_vec = np.zeros(localDim**2, dtype=complex)
    trace_vec[::localDim+1] = 1    # set diagonal elements to 1
    trace_vectors = [trace_vec for _ in range(nosc)]
    trace_assistant = qtn.MPS_product_state(trace_vectors)

    # Contract the MPS with the trace assistant MPS
    return complex(trace_assistant @ mps)
    
    
# Utility functions to construct local Hamiltonian
def local_ham_osc(omega, localDim):
    
    return omega * (np.kron(N_operator,np.eye(localDim))-np.kron(np.eye(localDim),N_operator.T))

# Utility functions to construct local dissipator
def local_dissipator(omega, temp, localDim):
    
    if temp == 0:
        n_bar = 0
    else:
        n_bar = 1 / (np.expm1(omega / temp))
        
    return (n_bar + 1) * (np.kron(a,np.conj(a)) - 0.5 * (np.kron(np.eye(localDim),N_operator.T) + np.kron(N_operator,np.eye(localDim)))) + n_bar * (np.kron(a_dag,np.conj(a_dag)) - 0.5 * (np.kron(np.eye(localDim),N1_operator.T) + np.kron(N1_operator,np.eye(localDim)))) 

def calculate_error(rho1, rho2, ns, nosc, localDim):
    partial_rho1 = partial_trace(rho1, ns, nosc, localDim)
    partial_rho2 = partial_trace(rho2, ns, nosc, localDim)
    return 0.5*np.abs(np.linalg.eigvalsh(partial_rho1-partial_rho2)).sum()

def partial_trace(rho, ns, nosc, localDim):
    
    partial_rho = np.zeros((ns, ns), dtype=complex)
    for i in range(ns):
        for j in range(ns):
            partial_rho[i][j] = trace_MPS(rho[i][j], nosc, localDim)
    return partial_rho