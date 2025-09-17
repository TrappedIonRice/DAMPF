'''
This file contains various utility functions used in the simulation
'''


import numpy as np
import scipy
import quimb.tensor as qtn


# Construct annihilation and creation operators for a single harmonic oscillator
def annihilation_operator(N, dtype=complex):
    
    a = np.zeros((N, N), dtype=dtype)
    # 0-based: a[n-1, n] = sqrt(n)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a

# Construct a 3D tensor whose elements are 1 if all three indices are equal, and 0 otherwise
def eye_3d(n):
    
    i, j, k = np.indices((n, n, n))
    T = (i == j) & (j == k)
    T = T.astype(int)
    return T

# Create initial states for oscillators for quantum trajectory method besed on the temperatures
def create_osc_initial_states(nosc, Ntraj, localDim, temps):
    
    osc_state_nparray = np.zeros((Ntraj, nosc), dtype=int)
    
    for i in range(nosc):
        if temps[i] == 0:
            osc_state_nparray[:, i] = 0
            continue
        
        # Attention: here temps[i] should be given in terms of nbar
        # Calculate the probability distribution for the oscillator states
        prob_list = np.array([(1 / temps[i]) * (temps[i] / (1 + temps[i])) ** (n+1) for n in range(localDim)])
        prob_list = prob_list / np.sum(prob_list)   # normalize the probability distribution
        osc_state_nparray[:, i] = np.random.choice(localDim, size=Ntraj, p=prob_list)
        
    return osc_state_nparray

'''
The evolution gates for electronic and oscillator parts are simply exponentials of the corresponding local Hamiltonians.
'''
def construct_onsite_gate(elham, nosc, freqs, a, a_dagger, dt):
        
    gate = [scipy.linalg.expm(-1j * dt * elham)]
    
    for i in range(nosc):
        local_gate = scipy.linalg.expm(-1j * dt * freqs[i] * a_dagger @ a)
        gate.append(local_gate)
    
    return qtn.MPO_product_operator(gate)

'''
The construction of interaction gates requires some derivation, which can be found in my illustrative notes of the whole project.
'''
def construct_interaction_gates(nsites, nosc, coups, a, a_dagger, dt):
    
    gates = []
    
    for i in range(nosc):
        
        T0 = eye_3d(nsites)
        T1 = np.array([scipy.linalg.expm(-1j * dt * coups[n][i] * (a + a_dagger)) for n in range(nsites)])
        
        M = np.tensordot(T0, T1, axes=(0, 0))
        M = M.transpose(1, 3, 0, 2)
        d0_in, d1_in, d0_out, d1_out = M.shape
        M_dense = M.reshape((d0_in * d1_in, d0_out * d1_out))
        
        submpo = qtn.MatrixProductOperator.from_dense(
            M_dense,
            dims=[d0_in, d1_in],   # physical dims for the two sites
            sites=[0, i + 1],
            L=nosc + 1
        )
        
        gates.append(submpo)
    
    return gates

'''
The non-unitary gates for the oscillators are exponentials of the non-Hermitian part of the effective Hamiltonian.
'''
def construct_on_site_non_unitary_gates(nosc, localDim, temps, freqs, damps, a, a_dagger, dt):
    
    gates = []
    nbar_array = temps
    
    for i in range(nosc):
    
        local_gate = scipy.linalg.expm(-0.5 * dt * damps[i] * ((1 + 2 * nbar_array[i]) * a_dagger @ a + nbar_array[i] * np.eye(localDim)))
        
        gates.append(local_gate)
    
    return gates

# Package all gates together for easy access
def construct_all_gates(nsites, elham, nosc, freqs, coups, temps, damps, localDim, dt):
    
    a = annihilation_operator(localDim)
    a_dagger = a.conj().T
    
    onsite_gate = construct_onsite_gate(elham, nosc, freqs, a, a_dagger, dt)
    interaction_gates = construct_interaction_gates(nsites, nosc, coups, a, a_dagger, dt)
    on_site_non_unitary_gates = construct_on_site_non_unitary_gates(nosc, localDim, temps, freqs, damps, a, a_dagger, dt)
    
    return onsite_gate, interaction_gates, on_site_non_unitary_gates

# Calculate the occupation number of each oscillator given the current state
def calculate_occupation_number(state, nosc, a, a_dagger):
    
    occupation_number = np.zeros(nosc)
    N_operator = a_dagger @ a
        
    terms = {(i,): N_operator for i in range(1, nosc + 1)}
    # Attention: the use of compute_local_expectation_canonical (which is more efficient than calculating expectation values one by one) requires the MPS to be properly canonicalized, and one of the feasible canonical form is complete right-canonical form.
    occupation_number = state.compute_local_expectation_canonical(terms=terms)
        
    return occupation_number.real

# Create initial thermal state as the initial state for the density matrix methods
def create_thermal_mps(nosc, localDim, temps, freqs):
    
    # List to store the single-site flattened density matrices
    rho_list = []
    
    for i in range(nosc):
        
        # Avoid division by zero for zero temperature
        if temps[i] == 0:
            rho = np.zeros((localDim, localDim))
            rho[0, 0] = 1.0
            rho_vec = rho.flatten()
            rho_list.append(rho_vec)
            continue
        
        # --- 1. Construct single oscillator thermal density matrix ---
        prob_array = np.array([(1 / temps[i]) * (temps[i] / (1 + temps[i])) ** (n+1) for n in range(localDim)])
        rho = np.diag(prob_array / np.sum(rho))   # normalize the density matrix
        
        # --- 2. Flatten the density matrix into a vector ---
        rho_vec = rho.flatten()
        rho_list.append(rho_vec)
        
    # --- 3. Create MPS from the list of flattened density matrices ---
    thermal_mps = qtn.MPS_product_state(rho_list)

    return thermal_mps

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
def local_ham_osc(omega, localDim, N_operator):
    
    return omega * (np.kron(N_operator,np.eye(localDim))-np.kron(np.eye(localDim),N_operator.T))

# Utility functions to construct local dissipator
def local_dissipator(omega, temp, localDim, a, a_dagger, N_operator, N1_operator):
    
    # Attention: here temp is essentially nbar
    return (temp + 1) * (np.kron(a,np.conj(a)) - 0.5 * (np.kron(np.eye(localDim),N_operator.T) + np.kron(N_operator,np.eye(localDim)))) + temp * (np.kron(a_dagger,np.conj(a_dagger)) - 0.5 * (np.kron(np.eye(localDim),N1_operator.T) + np.kron(N1_operator,np.eye(localDim))))
        

# Using the Frobenius norm of the difference between two reduced density matrices as the error measure for adaptive time-stepping
def calculate_error(rho1, rho2, ns, nosc, localDim):
    
    partial_rho1 = partial_trace(rho1, ns, nosc, localDim)
    partial_rho2 = partial_trace(rho2, ns, nosc, localDim)
    
    return np.linalg.norm(partial_rho1 - partial_rho2, 'fro')

# Utility function to perform partial trace over the oscillator degrees of freedom
def partial_trace(rho, ns, nosc, localDim):
    
    partial_rho = np.zeros((ns, ns), dtype=complex)
    for i in range(ns):
        for j in range(ns):
            partial_rho[i][j] = trace_MPS(rho[i][j], nosc, localDim)
            
    return partial_rho