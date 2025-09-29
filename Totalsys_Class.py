'''
This file contains the class definitions for the total system in different simulation methods, including:
1. Totalsys_Pure: Total system pure state with quantum trajectory method
2. Totalsys_Rho_Fixed_Step: Total system density matrix with fixed time step method
3. Totalsys_Rho_Adaptive_Step: Total system density matrix with adaptive time step method
'''


import numpy as np
import quimb.tensor as qtn
import utils
import scipy.linalg
from tqdm import tqdm


gates = {}

def init_gates(total_gates):
    # gates['onsite'] = onsite_gate
    # gates['interaction'] = interaction_gates
    # gates['on_site_non_unitary'] = on_site_non_unitary_gates
    gates['total_gates'] = total_gates


'''
The class Totalsys_Pure consists of information of the total system, including:

1. Parameters of the system (nsites, nosc, localDim, elham, freqs, coups, damps, temps, time, timestep, osc_state)
2. The total system pure state in MPS form, which consists of a spin site at the very left and nosc oscillator sites following it.
3. The population dynamics during time evolution (population), which will be updated after each time step by calculating the diagonal elements of the reduced density matrix of the electronic part, resulting in nsites numbers.

as well as methods for quantum trajectory time evolution and population update:

1. __init__: Initializes everything
2. Time_Evolve_Pure_QT: Time evolve the total system pure state with the help of various gates and quantum jumps
3. record_population: Update the population dynamics after each time step
4. various gates (as illustrated in the utils.py file)
'''

class Totalsys_Pure:
    
    def __init__(self, nsites, nosc, localDim, elham, freqs, coups, damps, temps, time, timestep, el_initial_state, osc_state, additional_osc_jump_op_dic={}, additional_osc_output_dic={}):
        
        self.maxbond_throughout_whole_evolution = 1
        
        # Parameter information initialization
        self.nsites = nsites
        self.nosc = nosc
        self.localDim = localDim
        self.elham = elham
        self.freqs = freqs
        self.coups = coups
        self.damps = damps
        self.temps = temps
        self.a = utils.annihilation_operator(localDim)
        self.a_dagger = self.a.conj().T
        
        # Gates initialization
        self.total_gates = gates['total_gates']
        # self.onsite_gate = gates['onsite']
        # self.interaction_gates = gates['interaction']
        # self.on_site_non_unitary_gates = gates['on_site_non_unitary']
        
        # State initialization
        self.el_initial_state = el_initial_state
        self.initialize_state(osc_state)
        
        # Population initialization
        self.results = {
            "reduced_density_matrix": np.zeros((nsites, nsites, int(time/timestep)), dtype=complex),
            "additional_osc_output": np.array([np.zeros(int(time/timestep), dtype=complex) for _ in range(len(additional_osc_output_dic))])
        }
        
        # Additional Requirements initialization
        self.additional_osc_jump_op_dic = additional_osc_jump_op_dic
        self.additional_osc_output_dic = additional_osc_output_dic
        
        # --- performance caches for repeated expectation calls (precompute terms) ---
        # occupation terms for compute_local_expectation_canonical:
        # keys are (site_index,) where oscillator sites are indexed from 1..nosc in your code
        self._occupation_terms = {(i,): self.a_dagger @ self.a for i in range(1, self.nosc + 1)}

        # additional jumps/outputs: create ordered lists and precompute terms used for expectation/probability
        self._additional_keys_jump = []
        self._additional_ops_jump = []
        self._additional_prob_terms_jump = {}        # for probabilities: op^† op
        self._additional_keys_output = []
        self._additional_ops_output = []
        self._additional_expectation_terms_output = {} # for expectation: op
        for key, op in additional_osc_jump_op_dic.items():
            ik = int(key)                       # ensure integer site index
            self._additional_keys_jump.append(ik)
            self._additional_ops_jump.append(op)
            self._additional_prob_terms_jump[(ik,)] = op.conj().T @ op
            
        for key, op in additional_osc_output_dic.items():
            ik = int(key)                       # ensure integer site index
            self._additional_keys_output.append(ik)
            self._additional_ops_output.append(op)
            self._additional_expectation_terms_output[(ik,)] = op

        self._n_additional_jump = len(self._additional_keys_jump)
        self._n_additional_output = len(self._additional_keys_output)
        

        
    def initialize_state(self, osc_state):
        
        if np.abs(np.linalg.norm(self.el_initial_state) - 1) > 1e-6:
            print("The electronic initial state is not normalized! Using the normalized state vector instead")
            
        self.el_initial_state = self.el_initial_state / np.linalg.norm(self.el_initial_state)
        init_el_state = [self.el_initial_state]
        init_osc_state = [np.eye(self.localDim)[excitation] for excitation in osc_state]
        self.state = qtn.MPS_product_state(init_el_state + init_osc_state)
        
    def Time_Evolve_Pure_QT(self, dt, total_time, maxBondDim):
        
        nsteps = int(total_time/dt)
        
        for step in range(nsteps):
            
            self.state.right_canonize()
            '''
            In Quantum Trajectory method, the tiny probability of quantum jumps is determined by the formula:
            delta_p_m = <psi| C_m^\dagger C_m |psi> * dt,
            where in our case, C_m is either sqrt(gamma_i * (nbar_i + 1)) * a_i or sqrt(gamma_i * nbar_i) * a_i^\dagger, representing the jump operators for the i-th oscillator. Therefore,
            delta_p1_i = gamma_i * (nbar_i + 1) * <psi| N_i |psi> * dt
            delta_p2_i = gamma_i * nbar_i * <psi| (N_i + 1) |psi> * dt
            '''
            occ_dict = self.state.compute_local_expectation_canonical(terms=self._occupation_terms, return_all=True)
            # occ_dict is a dict keyed by (i,) in ascending order of the keys in self._occupation_terms
            occupation_number = np.array(list(occ_dict.values())).real    # shape (nosc,)
            probability_1 = self.damps * (1 + self.temps) * dt * occupation_number
            probability_2 = self.damps * self.temps * dt * (occupation_number + 1)
            if self._n_additional_jump > 0:
                addp_dict = self.state.compute_local_expectation_canonical(terms=self._additional_prob_terms_jump, return_all=True)
                probability_3 = dt * np.array(list(addp_dict.values())).real
            else:
                probability_3 = np.array([])
                
            probability = np.hstack((probability_1, probability_2, probability_3))
            delta_p = probability.sum()
            
            # --- sampling optimized: first test no-jump quickly ---
            u = np.random.rand()
            if u > delta_p:
                # no-jump
                self.state.gate_with_mpo(
                    self.total_gates,
                    method='zipup',
                    max_bond=maxBondDim,
                    cutoff=1e-8,
                    canonize=True,
                    normalize=True,
                    inplace=True
                )
            else:
                # there is a jump — pick which one via cumulative-sum search
                r = np.random.rand() * delta_p
                cum = np.cumsum(probability)
                idx = int(np.searchsorted(cum, r, side='right'))  # index in [0, len(probability)-1]

                if idx < 2 * self.nosc:
                    # standard oscillator a or a^\dagger jump
                    osc_index = idx % self.nosc
                    jump_type = idx // self.nosc  # 0 for a, 1 for a_dagger
                    jump_op = self.a if jump_type == 0 else self.a_dagger

                    self.state.gate(
                        G=jump_op,
                        where=osc_index + 1,
                        inplace=True,
                        contract=True
                    )
                    self.state.normalize()
                    self.state.right_canonize()
                else:
                    # additional jump: use precomputed keys/ops lists
                    add_index = idx - 2 * self.nosc
                    osc_index = int(self._additional_keys_jump[add_index])
                    jump_op = self._additional_ops_jump[add_index]

                    self.state.gate(
                        G=jump_op,
                        where=osc_index,
                        inplace=True,
                        contract=True
                    )
                    self.state.normalize()
                    self.state.right_canonize()

            # record results as before
            self.record_results(step)
            
            if self.state.max_bond() > self.maxbond_throughout_whole_evolution:
                self.maxbond_throughout_whole_evolution = self.state.max_bond()
            
    def record_results(self, step):

        # electronic reduced density matrix
        T = self.state[0].data
        reduced_rho = T.T @ T.conj()
        assert reduced_rho.shape == (self.nsites, self.nsites), "Reduced density matrix shape error!"
        self.results["reduced_density_matrix"][:, :, step] = reduced_rho

        # additional outputs (use precomputed expectation terms to avoid rebuilding dicts)
        if self._n_additional_output > 0:
            add_expect = self.state.compute_local_expectation_canonical(terms=self._additional_expectation_terms_output, return_all=True)
            self.results["additional_osc_output"][:, step] = np.array(list(add_expect.values())).real
        else:
            # nothing to do (already initialized)
            pass

        
        
'''
The class Totalsys_Rho_Fixed_Step consists of information of the total system, including:

1. Parameters of the system (nsites, nosc, localDim, temps, freqs, damps, coups, time, timestep, elham)
2. The total system density matrix in MPS form (rho), which is a 2D array of shape (nsites, nsites), with each element being an MPS (flattened MPO) representing the density matrices of the oscillators
3. The population dynamics during time evolution (populations), which will be updated after each time step by tracing out the diagonal MPS elements, resulting in nsites numbers,

as well as methods for fixed step time evolution, population update, and construction of various evolution gates:

1. __init__: Initializes everything
2. Time_Evolve_Rho_Fixed_Step: Time evolve the total system density matrix with the help of various gates
3. update_populations: Update the population dynamics after each time step
4. various gates (as illustrated below)
'''

class Totalsys_Rho_Fixed_Step:
    
    def __init__(self, nsites, nosc, localDim, temps, freqs, damps, coups, time, timestep, elham, el_initial_state):
        
        thermal_mps = utils.create_thermal_mps(nosc, localDim, temps, freqs)
        
        # The zero_mps should be in the same structure as the thermal_mps, otherwise it will cause an error when adding two different-structured MPSs.
        zero_mps = thermal_mps.copy()
        for t in zero_mps.tensors:
            t.modify(data=t.data*0)
            
        if np.abs(np.linalg.norm(el_initial_state) - 1) > 1e-6:
            print("The electronic initial state is not normalized! Using the normalized state vector instead")
        
        self.el_initial_state = el_initial_state / np.linalg.norm(el_initial_state)
        
        # Construct the initial density matrix rho
        self.rho = np.outer(self.el_initial_state, np.conjugate(self.el_initial_state)) * thermal_mps
        
        # Population initialization
        self.populations = np.zeros((nsites, int(time/timestep)))
        
        # Parameter information initialization
        self.nsites = nsites
        self.localDim = localDim
        self.nosc = nosc
        self.temps = temps
        self.freqs = freqs
        self.damps = damps
        self.coups = coups
        self.elham = elham
        self.a = utils.annihilation_operator(localDim)
        self.a_dagger = self.a.conj().T
        self.N_operator = self.a_dagger @ self.a
        self.N1_operator = self.a @ self.a_dagger
        
    def Time_Evolve_Rho_Fixed_Step(self, time, dt, max_bond_dim):
        
        '''
        The Total system is described by three parts of Hamiltonian.
        1. The electronic Hamiltonian (H_el):
            H_el = sum_n E_n |n><n| + sum_{m!=n} J_{mn} |m><n|
        2. The oscillator Hamiltonian (H_osc):
            H_osc = sum_i omega_i (a_i^\dagger a_i + 1/2)
        3. The interaction Hamiltonian (H_int):
            H_int = sum_n sum_i (g_{n,i} |n><n| (a_i + a_i^\dagger))
        with dissipators for each oscillator:
            D_i(rho) = gamma_i (nbar_i + 1) (a_i rho a_i^\dagger - 1/2 {a_i^\dagger a_i, rho}) + gamma_i nbar_i (a_i^\dagger rho a_i - 1/2 {a_i a_i^\dagger, rho})
        where nbar_i = 1/(exp(beta omega_i) - 1) is the thermal occupation number of the i-th oscillator.
        
        These three Hamiltonians will be used to calculate the commutator [H, rho], from which we can get the evolution superoperators for a small time step dt. These superoperators, together with the superoperators from the dissipators, will be used to time evolve the flattened density matrix in MPS form.
        '''
        
        # Two evolution gates
        osc_gates = self.get_osc_gates(dt/2)
        int_gates = self.get_int_gates(dt/2)

        # Electronic evolution Coefficients       
        U_el = scipy.linalg.expm(-1j * dt * self.elham)
        U_el_dagger = U_el.conj().T
        
        # Main time evolution loop (using 2nd order Trotter decomposition)
        for step in tqdm(range(int(time/dt))):
            
            ns = self.nsites

            # Oscillator part and interaction part evolution with dt/2
            for i in range(ns):
                for j in range(ns):
                    if i <= j:
                        self.rho[i][j] = self.rho[i][j].gate_with_mpo(
                            osc_gates,
                            method='zipup',
                            max_bond=max_bond_dim,
                            cutoff=1e-8,
                        )
                        self.rho[i][j] = self.rho[i][j].gate_with_mpo(
                            int_gates[i][j],
                            method='zipup',
                            max_bond=max_bond_dim,
                            cutoff=1e-8,
                        )
                    else:
                        # Use the Hermitian property of the density matrix to reduce computation
                        self.rho[i][j] = self.rho[j][i].conj()

            self.rho = np.dot(U_el, np.dot(self.rho, U_el_dagger))
            
            # Interaction part and oscillator part evolution with dt/2
            for i in range(ns):
                for j in range(ns):
                    if i <= j:
                        self.rho[i][j] = self.rho[i][j].gate_with_mpo(
                            int_gates[i][j],
                            method='zipup',
                            max_bond=max_bond_dim,
                            cutoff=1e-8,
                        )
                        self.rho[i][j] = self.rho[i][j].gate_with_mpo(
                            osc_gates,
                            method='zipup',
                            max_bond=max_bond_dim,
                            cutoff=1e-8,
                        )
                    else:
                        # Use the Hermitian property of the density matrix to reduce computation
                        self.rho[i][j] = self.rho[j][i].conj()
            
            self.update_populations(step, ns)
            
    def update_populations(self, step, ns):
        
        for n in range(ns):
            self.populations[n][step] = utils.trace_MPS(self.rho[n][n], self.nosc, self.localDim).real
    
    # The evolution gates of local oscillators are identical among all matrix elements (matrix elements refer to the elements in the total system density matrix), so we only return one MPO here.
    def get_osc_gates(self, dt):
        
        local_ops = []
        for i in range(self.nosc):
            local_ops.append(scipy.linalg.expm(-1j * dt * utils.local_ham_osc(self.freqs[i], self.localDim, self.N_operator) + dt * self.damps[i] * utils.local_dissipator(self.freqs[i], self.temps[i], self.localDim, self.a, self.a_dagger, self.N_operator, self.N1_operator)))
        
        # Combine local operators into an MPO, with bond dimensions of 1   
        return qtn.MPO_product_operator(local_ops)
    
    # Conversely, the interaction gates are different among different matrix elements, so we return a 2D array of MPOs here.
    
    def get_int_gates(self, dt):
        X = (self.a + self.a_dagger)              # shape (localDim, localDim)
        X_t = X.T
        ns = self.nsites
        no = self.nosc
        expm_minus = [[None]*no for _ in range(ns)]   # exp(-i dt g_m X)
        expm_plus  = [[None]*no for _ in range(ns)]   # exp( i dt g_n X^T)
        
        for m in range(ns):
            for i in range(no):
                g = self.coups[m][i]
                expm_minus[m][i] = scipy.linalg.expm(-1j * dt * g * X)
                expm_plus[m][i]  = scipy.linalg.expm( 1j * dt * g * X_t)

        int_gates = np.empty((ns, ns), dtype=object)
        
        for m in range(ns):
            for n in range(ns):
                local_ops = []
                for i in range(no):
                    local_ops.append(np.kron(expm_minus[m][i], expm_plus[n][i]))
                int_gates[m][n] = qtn.MPO_product_operator(local_ops)
                
        return int_gates


'''
The class Totalsys_Rho_Adaptive_Step consists of information of the total system, including:

1. Parameters of the system (self, nsites, nosc, localDim, elham, temps, freqs, damps, coups, dt_array)
2. The total system density matrix in MPS form (rho), which is a 2D array of shape (nsites, nsites), with each element being an MPS (flattened MPO) representing the density matrices of the oscillators
3. The population dynamics during time evolution (populations), which will be updated after each time step by tracing out the diagonal MPS elements, resulting in nsites numbers,

as well as methods for time evolution, population update, and construction of various evolution gates:

1. __init__: Initializes everything
2. Time_Evolve_Rho_Adaptive_Step: Time evolve the total system density matrix with the help of various gates
3. specific_time_evolve: Perform a specific time evolution with a given dt index and indicator (0 or 1), where indicator 0 means one application of the whole value dt, and indicator 1 means two applications of half value dt
4. update_populations: Update the population dynamics after each time step
5. various gates (as illustrated below)
'''

class Totalsys_Rho_Adaptive_Step:
    
    def __init__(self, nsites, nosc, localDim, elham, temps, freqs, damps, coups, dt_array, el_initial_state):
        
        thermal_mps = utils.create_thermal_mps(nosc, localDim, temps, freqs)
        
        # The zero_mps should be in the same structure as the thermal_mps, otherwise it will cause an error when adding two different-structured MPSs.
        zero_mps = thermal_mps.copy()
        for t in zero_mps.tensors:
            t.modify(data=t.data*0)
        
        if np.abs(np.linalg.norm(el_initial_state) - 1) > 1e-6:
            print("The electronic initial state is not normalized! Using the normalized state vector instead")
        
        self.el_initial_state = el_initial_state / np.linalg.norm(el_initial_state)
        
        # Construct the initial density matrix rho
        self.rho = np.outer(self.el_initial_state, np.conjugate(self.el_initial_state)) * thermal_mps
        
        # Population initialization
        self.populations = [[] for _ in range(nsites)]
        
        # Parameter information initialization
        self.nsites = nsites
        self.localDim = localDim
        self.nosc = nosc
        self.temps = temps
        self.freqs = freqs
        self.damps = damps
        self.coups = coups
        self.elham = elham
        self.a = utils.annihilation_operator(localDim)
        self.a_dagger = self.a.conj().T
        self.N_operator = self.a_dagger @ self.a
        self.N1_operator = self.a @ self.a_dagger
        
        self.dt_array = dt_array
        
        # Pre-construct various evolution gates for all possible discrete dt values in dt_array
        print("Constructing various evolution gates...")
        print("It may take a while, please be patient.")
        self.osc_gates_dt_div_2 = [[self.get_osc_gates(dt/2) for dt in dt_row] for dt_row in dt_array]
        print("Oscillator gates constructed.")
        self.int_gates_dt_div_2 = [[self.get_int_gates(dt/2) for dt in dt_row] for dt_row in dt_array]
        print("Interaction gates constructed.")
        self.U_els_dt_div_1 = [[self.get_U_els(dt) for dt in dt_row] for dt_row in dt_array]
        print("Electronic evolution operators constructed.")

        
    def Time_Evolve_Rho_Adaptive_Step(self, total_time, initial_dt, max_bond_dim, err_tol, S1, S2):
        
        '''
        The Total system is described by three parts of Hamiltonian.
        1. The electronic Hamiltonian (H_el):
            H_el = sum_n E_n |n><n| + sum_{m!=n} J_{mn} |m><n|
        2. The oscillator Hamiltonian (H_osc):
            H_osc = sum_i omega_i (a_i^\dagger a_i + 1/2)
        3. The interaction Hamiltonian (H_int):
            H_int = sum_n sum_i (g_{n,i} |n><n| (a_i + a_i^\dagger))
        with dissipators for each oscillator:
            D_i(rho) = gamma_i (nbar_i + 1) (a_i rho a_i^\dagger - 1/2 {a_i^\dagger a_i, rho}) + gamma_i nbar_i (a_i^\dagger rho a_i - 1/2 {a_i a_i^\dagger, rho})
        where nbar_i = 1/(exp(beta omega_i) - 1) is the thermal occupation number of the i-th oscillator.
        
        These three Hamiltonians will be used to calculate the commutator [H, rho], from which we can get the evolution superoperators for a small time step dt. These superoperators, together with the superoperators from the dissipators, will be used to time evolve the flattened density matrix in MPS form.
        '''
        
        current_time = 0.0
        dt = initial_dt
        step = 0
        Time = []
        
        # Main time evolution loop (using adaptive time step method based on the order-2 Suzuki-Trotter method)
        with tqdm(total=total_time, desc="Integrating", unit="t") as pbar:
            
            while current_time < total_time:
                
                ns = self.nsites
                nosc = self.nosc
                localDim = self.localDim
                
                # The last time step
                if current_time + dt > total_time:
                    dt = total_time - current_time
                    pbar.update(dt)
                    current_time += dt
                    break
                    
                if np.all(self.dt_array[0] > dt):
                    print("Warning: The current time step is smaller than the minimum value in dt_array[0].")
                    print("Please consider using a modified dt_array.")
                    exit()
                elif np.all(self.dt_array[0] < dt):
                    print("Warning: The current time step is larger than the maximum value in dt_array[0].")
                
                # Find the closest dt in dt_array that is smaller than or equal to the current dt value, and get its index
                dt = max(self.dt_array[0][self.dt_array[0] <= dt], default=None)
                dt_index = np.where(self.dt_array[0] == dt)[0][0]
                
                # Perform one evolution with dt (indicator=0) and two evolutions with dt/2 (indicator=1), then compare the results to estimate the local error    
                rho0 = self.rho.copy()
                rho1 = self.specific_time_evolve(rho0.copy(), dt_index, 0, max_bond_dim)
                rho_temp = self.specific_time_evolve(rho0.copy(), dt_index, 1, max_bond_dim)
                rho2 = self.specific_time_evolve(rho_temp.copy(), dt_index, 1, max_bond_dim)
                
                err = utils.calculate_error(rho1, rho2, ns, nosc, localDim)
                
                # Estimate the new time step based on the local error
                if err == 0:
                    dt_est = dt * S2
                else:
                    dt_est = dt * abs(err_tol/err)**(1/3)
                
                # Further limit the change of time step
                dt_new = S1 * dt_est
                if dt_new > S2 * dt:
                    dt_new = S2 * dt
                elif dt_new < dt / S2:
                    dt_new = dt / S2
                
                # Accept the current step if the local error is smaller than the error tolerance
                if err < err_tol:
                    
                    Time.append(current_time)
                    pbar.update(dt)
                    current_time += dt
                    self.rho = rho2
                    self.update_populations(step, ns)
                    dt = dt_new
                    step += 1
                
                # Otherwise reject it and redo the step with the new time step
                else:
                    
                    print(f"Step {step}: Rejected due to large local error {err:.2e} > {err_tol:.2e}. Redoing with dt = {dt_new:.2e}.")
                    dt = dt_new
                
        return Time
            
    def update_populations(self, step, ns):
        
        for n in range(ns):
            self.populations[n].append(utils.trace_MPS(self.rho[n][n], self.nosc, self.localDim).real)
            
    '''
    This specific_time_evolve function performs time evolution using order-2 Suzuki-Trotter method, the formula expressed via superoperators is as follows:
    rho(t+dt) = e^{(L_osc + D_osc) dt/2} e^{L_int dt/2} e^{L_el dt} e^{L_int dt/2} e^{(L_osc + D_osc) dt/2} rho(t)
    '''
    def specific_time_evolve(self, rho, dt_index, indicator, max_bond_dim):
        
        ns = self.nsites
        
        # Two evolution gates
        osc_gates = self.osc_gates_dt_div_2[indicator][dt_index]
        int_gates = self.int_gates_dt_div_2[indicator][dt_index]

        # Electronic evolution Matrix
        U_el = self.U_els_dt_div_1[indicator][dt_index]
        U_el_dagger = U_el.conj().T
        
        result_rho = rho
            
        # Oscillator part and interaction part evolution with dt/2
        for i in range(ns):
            for j in range(ns):
                if i <= j:
                    result_rho[i][j] = result_rho[i][j].gate_with_mpo(
                        osc_gates,
                        method='zipup',
                        max_bond=max_bond_dim,
                        cutoff=1e-8,
                    )
                    result_rho[i][j] = result_rho[i][j].gate_with_mpo(
                        int_gates[i][j],
                        method='zipup',
                        max_bond=max_bond_dim,
                        cutoff=1e-8,
                    )
                else:
                    # Use the Hermitian property of the density matrix to reduce computation
                    result_rho[i][j] = result_rho[j][i].conj()

        # Electronic part evolution
        result_rho = np.dot(U_el, np.dot(result_rho, U_el_dagger))
        
        # Interaction part and oscillator part evolution with dt/2
        for i in range(ns):
            for j in range(ns):
                if i <= j:
                    result_rho[i][j] = result_rho[i][j].gate_with_mpo(
                        int_gates[i][j],
                        method='zipup',
                        max_bond=max_bond_dim,
                        cutoff=1e-8,
                    )
                    result_rho[i][j] = result_rho[i][j].gate_with_mpo(
                        osc_gates,
                        method='zipup',
                        max_bond=max_bond_dim,
                        cutoff=1e-8,
                    )
                else:
                    # Use the Hermitian property of the density matrix to reduce computation
                    result_rho[i][j] = result_rho[j][i].conj()
                
        return result_rho
                
            
    # The evolution gates of local oscillators are identical among all matrix elements (matrix elements refer to the elements in the total system density matrix), so we only return one MPO here.
    def get_osc_gates(self, dt):
        
        local_ops = []
        for i in range(self.nosc):
            local_ops.append(scipy.linalg.expm(-1j * dt * utils.local_ham_osc(self.freqs[i], self.localDim, self.N_operator) + dt * self.damps[i] * utils.local_dissipator(self.freqs[i], self.temps[i], self.localDim, self.a, self.a_dagger, self.N_operator, self.N1_operator)))
        
        # Combine local operators into an MPO, with bond dimensions of 1   
        return qtn.MPO_product_operator(local_ops)
    
    # Conversely, the interaction gates are different among different matrix elements, so we return a 2D array of MPOs here.
    def get_int_gates(self, dt):
        
        X = (self.a + self.a_dagger)              # shape (localDim, localDim)
        X_t = X.T
        ns = self.nsites
        no = self.nosc
        expm_minus = [[None]*no for _ in range(ns)]   # exp(-i dt g_m X)
        expm_plus  = [[None]*no for _ in range(ns)]   # exp( i dt g_n X^T)
        
        for m in range(ns):
            for i in range(no):
                g = self.coups[m][i]
                expm_minus[m][i] = scipy.linalg.expm(-1j * dt * g * X)
                expm_plus[m][i]  = scipy.linalg.expm( 1j * dt * g * X_t)

        int_gates = np.empty((ns, ns), dtype=object)
        for m in range(ns):
            for n in range(ns):
                local_ops = []
                for i in range(no):
                    local_ops.append(np.kron(expm_minus[m][i], expm_plus[n][i]))
                int_gates[m][n] = qtn.MPO_product_operator(local_ops)
                
        return int_gates
    
    # The electronic evolution operator is simply a matrix of complex numbers
    def get_U_els(self, dt):
        
        return scipy.linalg.expm(-1j * dt * self.elham)