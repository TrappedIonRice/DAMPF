'''
This file contains various utility functions used in the simulation
'''


import numpy as np
import scipy
import quimb.tensor as qtn










'''
--- Functions shared by all three classes ---
'''
# Construct annihilation and creation operators for a single harmonic oscillator
def annihilation_operator(N, dtype=complex):
    
    a = np.zeros((N, N), dtype=dtype)
    # 0-based: a[n-1, n] = sqrt(n)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a










'''
--- Functions specific to the Pure_QT Method ---
'''
# Construct a 3D tensor whose elements are 1 if all three indices are equal, and 0 otherwise
def eye_3d(n):
    
    i, j, k = np.indices((n, n, n))
    T = (i == j) & (j == k)
    T = T.astype(int)
    return T

# Create initial states for oscillators for quantum trajectory method besed on the temperatures
def create_thermal_osc_initial_states(nosc, Ntraj, localDim, temps):
    
    osc_state_nparray = np.zeros((Ntraj, nosc, localDim), dtype=complex)
    
    for i in range(nosc):
        if temps[i] == 0:
            osc_state_nparray[:, i, :] = np.eye(localDim)[np.array([0] * Ntraj)]
            continue
        
        # Attention: here temps[i] should be given in terms of nbar
        # Calculate the probability distribution for the oscillator states
        prob_list = np.array([(1 / (temps[i]+1)) * (temps[i] / (1 + temps[i])) ** n for n in range(localDim)])
        prob_list = prob_list / np.sum(prob_list)   # normalize the probability distribution
        osc_state_nparray[:, i, :] = np.eye(localDim)[np.random.choice(localDim, size=Ntraj, p=prob_list)]
        
    return osc_state_nparray

# The evolution gates for electronic and oscillator parts are simply exponentials of the corresponding local Hamiltonians.
def construct_onsite_gate(elham, nosc, freqs, a, a_dagger, dt):
        
    gate = [scipy.linalg.expm(-1j * dt * elham)]
    
    for i in range(nosc):
        local_gate = scipy.linalg.expm(-1j * dt * freqs[i] * a_dagger @ a)
        gate.append(local_gate)
    
    return qtn.MPO_product_operator(gate)

# The construction of interaction gates requires some derivation, which can be found in the illustrative notes of the whole project.
def construct_interaction_gates(nsites, nosc, localDim, coups, a, a_dagger, dt):
    
    gates = []
    
    for i in range(nosc):
        
        T0 = eye_3d(nsites)
        T1 = np.array([scipy.linalg.expm(-1j * dt / 2 * coups[n][i] * (a + a_dagger)) for n in range(nsites)])
        
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
        
        fullmpo = fill_sites(submpo, nsites, nosc, localDim, second_index=i + 1)
        # fullmpo = submpo.fill_empty_sites(phys_dim=localDim, inplace=False)
        
        gates.append(fullmpo)
    
    return gates

def fill_sites(submpo, nsites, nosc, localDim, second_index):
    
    bond_dim = submpo[0].data.shape[2]
    temp_tensor = np.transpose(submpo[0].data,(2, 0, 1))
    tensor_array = [np.expand_dims(temp_tensor, axis=0)]
    
    for i in range(1, 1+nosc):
        if i == second_index:
            temp_tensor = submpo[i].data
            tensor_array.append(np.expand_dims(temp_tensor, axis=1))
            bond_dim = 1
        else:
            tensor_array.append(np.einsum('ij,kl->ijkl', np.eye(bond_dim), np.eye(localDim)))
    
    fullmpo = qtn.MatrixProductOperator(tensor_array, shape='lrud')
    return fullmpo

# The non-unitary gates for the oscillators are exponentials of the non-Hermitian part of the effective Hamiltonian.
def construct_on_site_non_unitary_gates(nsites, nosc, localDim, temps, freqs, damps, a, a_dagger, dt):
    
    gates = [np.eye(nsites)]
    nbar_array = temps
    
    for i in range(nosc):
    
        local_gate = scipy.linalg.expm(-0.5 * dt * damps[i] * ((1 + 2 * nbar_array[i]) * a_dagger @ a + nbar_array[i] * np.eye(localDim)))
        
        gates.append(local_gate)
    
    return qtn.MPO_product_operator(gates)

# Package all gates together for easy access
def construct_all_gates(nsites, elham, nosc, freqs, coups, temps, damps, localDim, dt):
    
    a = annihilation_operator(localDim)
    a_dagger = a.conj().T
    
    onsite_gate = construct_onsite_gate(elham, nosc, freqs, a, a_dagger, dt)
    interaction_gates = construct_interaction_gates(nsites, nosc, localDim, coups, a, a_dagger, dt)
    on_site_non_unitary_gates = construct_on_site_non_unitary_gates(nsites, nosc, localDim, temps, freqs, damps, a, a_dagger, dt)
    
    total_gates_np = [np.eye(nsites)] + [np.eye(localDim) for _ in range(nosc)]
    total_gates = qtn.MPO_product_operator(total_gates_np)
    for gate in interaction_gates:
        total_gates = qtn.tensor_network_apply_op_op(
            total_gates, gate, 
            which_A='lower', which_B='upper', 
            contract=True, fuse_multibonds=True, compress=True, 
            inplace=False, inplace_A=False, 
            max_bond=100, cutoff=1e-12, method='svd'
        )
    total_gates = qtn.tensor_network_apply_op_op(
        total_gates, onsite_gate, 
        which_A='lower', which_B='upper',
        contract=True, fuse_multibonds=True, compress=True,
        inplace=False, inplace_A=False,
        max_bond=100, cutoff=1e-12, method='svd'
    )
    total_gates = qtn.tensor_network_apply_op_op(
        total_gates, on_site_non_unitary_gates,
        which_A='lower', which_B='upper',
        contract=True, fuse_multibonds=True, compress=True,
        inplace=False, inplace_A=False,
        max_bond=100, cutoff=1e-12, method='svd'
    )
    for gate in interaction_gates:
        total_gates = qtn.tensor_network_apply_op_op(
            total_gates, gate, 
            which_A='lower', which_B='upper', 
            contract=True, fuse_multibonds=True, compress=True, 
            inplace=False, inplace_A=False, 
            max_bond=100, cutoff=1e-12, method='svd'
        )
    
    return total_gates

# Calculate the occupation number of each oscillator given the current state
def calculate_occupation_number(state, nosc, a, a_dagger):
    
    occupation_number = np.zeros(nosc, dtype=complex)
    N_operator = a_dagger @ a
    
    terms = {(i,): N_operator for i in range(1, nosc + 1)}
    # Attention: the use of compute_local_expectation_canonical (which is more efficient than calculating expectation values one by one) requires the MPS to be properly canonicalized, and one of the feasible canonical form is complete right-canonical form.
    occupation_number = state.compute_local_expectation_canonical(terms=terms, return_all=True)
    occupation_number_nparray = np.array(list(occupation_number.values()))
    # print(occupation_number_nparray)
    return occupation_number_nparray.real

def calculate_additional_probabilities(state, additional_output_dic):
    
    if len(additional_output_dic) == 0:
        return np.array([])
    
    terms = {tuple([int(key)]): op.conj().T @ op for key, op in additional_output_dic.items()}
    # Attention: the use of compute_local_expectation_canonical (which is more efficient than calculating expectation values one by one) requires the MPS to be properly canonicalized, and one of the feasible canonical form is complete right-canonical form.
    additional_probabilities = state.compute_local_expectation_canonical(terms=terms, return_all=True)
    
    return np.array(list(additional_probabilities.values())).real
    
def calculate_additional_expectation(state, additional_output_dic):
    
    if len(additional_output_dic) == 0:
        return np.array([])
    
    terms = {tuple([int(key)]): op for key, op in additional_output_dic.items()}
    # Attention: the use of compute_local_expectation_canonical (which is more efficient than calculating expectation values one by one) requires the MPS to be properly canonicalized, and one of the feasible canonical form is complete right-canonical form.
    additional_expectation = state.compute_local_expectation_canonical(terms=terms, return_all=True)
    
    return np.array(list(additional_expectation.values())).real










'''
--- Functions specific to the Density Matrix Method ---
'''
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
        prob_array = np.array([(1 / (temps[i]+1)) * (temps[i] / (1 + temps[i])) ** n for n in range(localDim)])
        rho = np.diag(prob_array / np.sum(prob_array))   # normalize the density matrix
        
        # --- 2. Flatten the density matrix into a vector ---
        rho_vec = rho.flatten()
        rho_list.append(rho_vec)
        
    # --- 3. Create MPS from the list of flattened density matrices ---
    thermal_mps = qtn.MPS_product_state(rho_list)

    return thermal_mps

# The trace of an MPS (sitewisely flattened MPO) cannot be calculated directly using the built-in function in quimb, so we define our own function here.
# We are calculating the trace by contracting the MPS with a series of identical vectors that pick out the diagonal elements.
# module-level cache
_TRACE_MPS_CACHE = {}

def trace_MPS(mps, nosc, localDim):
    global _TRACE_MPS_CACHE
    cache_key = (nosc, localDim)
    if cache_key in _TRACE_MPS_CACHE:
        trace_assistant = _TRACE_MPS_CACHE[cache_key]
    else:
        trace_vec = np.zeros(localDim**2, dtype=complex)
        trace_vec[::localDim+1] = 1
        trace_vectors = [trace_vec for _ in range(nosc)]
        trace_assistant = qtn.MPS_product_state(trace_vectors)
        _TRACE_MPS_CACHE[cache_key] = trace_assistant

    # Contract the MPS with the trace assistant (no need to rebuild the assistant each call)
    return complex(trace_assistant @ mps)

    
# Utility functions to construct local Hamiltonian
def local_ham_osc(omega, localDim, N_operator):
    
    return omega * (np.kron(N_operator,np.eye(localDim))-np.kron(np.eye(localDim),N_operator.T))

def local_dissipator(omega, nbar, localDim, a, a_dag, N_operator=None, N1_operator=None):
    """
    Row-major (NumPy default) vec convention:
      vec_C(A @ rho @ B) = (A ⊗ B.T) vec_C(rho)

    Lindblad dissipator for a single bosonic mode:
      D[rho] = (nbar+1)*( a rho a^dag - 1/2 {a^dag a, rho} )
             +  nbar   *( a^dag rho a - 1/2 {a a^dag, rho} )

    Returns a superoperator matrix of shape (d^2, d^2).
    """
    # Ensure complex dtype
    a = np.asarray(a, dtype=complex)
    a_dag = np.asarray(a_dag, dtype=complex)
    d = localDim
    I = np.eye(d, dtype=complex)

    # number operators if not passed
    if N_operator is None:
        N_operator = a_dag @ a
    if N1_operator is None:
        N1_operator = a @ a_dag

    # emission term: a rho a^\dagger  -> np.kron(a, a_dag.T)
    L_em_jump = np.kron(a, a_dag.T)
    # absorption term: a^\dagger rho a -> np.kron(a_dag, a.T)
    L_ab_jump = np.kron(a_dag, a.T)

    # anti-commutator parts:
    # For emission anti-commutator {a^\dagger a, rho} => (N ⊗ I + I ⊗ N.T)
    anti_em = 0.5 * (np.kron(N_operator, I) + np.kron(I, N_operator.T))
    # For absorption anti-commutator {a a^\dagger, rho}
    anti_ab = 0.5 * (np.kron(N1_operator, I) + np.kron(I, N1_operator.T))

    L = (nbar + 1) * (L_em_jump - anti_em) + nbar * (L_ab_jump - anti_ab)

    return L

# # Utility functions to construct local dissipator
# def local_dissipator(omega, temp, localDim, a, a_dagger, N_operator, N1_operator):
    
#     # Attention: here temp is essentially nbar
#     return (temp + 1) * (np.kron(a,np.conj(a)) - 0.5 * (np.kron(np.eye(localDim),N_operator.T) + np.kron(N_operator,np.eye(localDim)))) + temp * (np.kron(a_dagger,np.conj(a_dagger)) - 0.5 * (np.kron(np.eye(localDim),N1_operator.T) + np.kron(N1_operator,np.eye(localDim))))
        

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