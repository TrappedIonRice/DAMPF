'''
This module defines the Totalsys_Rho class, which is the core part of this whole project.
'''


import numpy as np
import DAMPF.utils
import quimb.tensor as qtn
import DAMPF.config
from tqdm import tqdm
import scipy.linalg


a = DAMPF.utils.annihilation_operator(DAMPF.config.localDim)
a_dag = a.conj().T


'''
The class Totalsys_Rho consists of information of the total system, including:

1. Parameters of the system (nsites, noscpersite, localDim, nosc, temps, freqs, damps, coups, energies, exchange, a, a_dag)
2. The total system density matrix in MPS form (rho), which is a 2D array of shape (nsites, nsites), with each element being an MPS (flattened MPO) representing the density matrices of the oscillators
3. The population dynamics during time evolution (populations), which will be updated after each time step by tracing out the diagonal MPS elements, resulting in nsites numbers,

as well as methods for time evolution, population update, and construction of various evolution gates:

1. __init__: Initializes everything
2. Time_Evolve: Time evolve the total system density matrix with the help of various gates
3. specific_time_evolve: Perform a specific time evolution with a given dt index and indicator (0 or 1), where indicator 0 means one application of the whole value dt, and indicator 1 means two applications of half value dt
4. update_populations: Update the population dynamics after each time step
5. Various gates (as illustrated below)

'''
class Totalsys_Rho:
    
    def __init__(self, nsites, noscpersite, nosc, localDim, temps, freqs, damps, coups, dt_array):
        
        thermal_mps = DAMPF.utils.create_thermal_mps(nosc, localDim, temps, freqs)
        
        # The zero_mps should be in the same structure as the thermal_mps, otherwise it will cause an error when adding two different-structured MPSs.
        zero_mps = thermal_mps.copy()
        for t in zero_mps.tensors:
            t.modify(data=t.data*0)
        
        # All of the elements in the intial rho is zero_mps, except the top-left corner
        self.rho = [[thermal_mps if (i==0) and (j==0) else zero_mps for i in range(nsites)] for j in range(nsites)]
        
        # Population initialization
        self.populations = [[] for _ in range(nsites)]
        # self.test_populations = np.zeros((nsites, int(DAMPF.config.time/DAMPF.config.timestep)))
        
        # Parameter information initialization
        self.nsites = nsites
        self.noscpersite = noscpersite
        self.localDim = localDim
        self.nosc = nosc
        self.temps = temps
        self.freqs = freqs
        self.damps = damps
        self.coups = coups
        self.energies = DAMPF.config.energies
        self.exchange = DAMPF.config.exchange
        self.a = a
        self.a_dag = a_dag
        self.el_ham = self.exchange + np.diag(self.energies)
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

        
    def Time_Evolve(self, total_time, initial_dt, max_bond_dim, err_tol, S1, S2):
        
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
                
                err = DAMPF.utils.calculate_error(rho1, rho2, ns, nosc, localDim)
                
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
            self.populations[n].append(DAMPF.utils.trace_MPS(self.rho[n][n], self.nosc, self.localDim).real)
            
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
        # U_el = scipy.linalg.expm(-1j * dt * self.el_ham)
        U_el = self.U_els_dt_div_1[indicator][dt_index]
        U_el_dag = U_el.conj().T
        
        result_rho = rho
            
        # Oscillator part and interaction part evolution with dt/2
        for i in range(ns):
            for j in range(ns):
                if i <= j:
                    result_rho[i][j] = result_rho[i][j].gate_with_mpo(osc_gates)
                    result_rho[i][j] = result_rho[i][j].gate_with_mpo(int_gates[i][j])
                else:
                    # Use the Hermitian property of the density matrix to reduce computation
                    result_rho[i][j] = result_rho[j][i].conj()

        # Electronic part evolution
        result_rho = np.dot(U_el, np.dot(result_rho, U_el_dag))
        
        # Interaction part and oscillator part evolution with dt/2
        for i in range(ns):
            for j in range(ns):
                if i <= j:
                    result_rho[i][j] = result_rho[i][j].gate_with_mpo(int_gates[i][j])
                    result_rho[i][j] = result_rho[i][j].gate_with_mpo(osc_gates)
                else:
                    # Use the Hermitian property of the density matrix to reduce computation
                    result_rho[i][j] = result_rho[j][i].conj()
        
        # Compress the MPS after each time step to control the bond dimension
        for i in range(ns):
            for j in range(ns):
                result_rho[i][j].compress(max_bond=max_bond_dim)
                
        return result_rho
                
            
    # The evolution gates of local oscillators are identical among all matrix elements (matrix elements refer to the elements in the total system density matrix), so we only return one MPO here.
    def get_osc_gates(self, dt):
        
        local_ops = []
        for i in range(self.nosc):
            local_ops.append(scipy.linalg.expm(-1j * dt * DAMPF.utils.local_ham_osc(self.freqs[i], self.localDim) + dt * self.damps[i] * DAMPF.utils.local_dissipator(self.freqs[i], self.temps[i], self.localDim)))
        
        # Combine local operators into an MPO, with bond dimensions of 1   
        return qtn.MPO_product_operator(local_ops)
    
    # Conversely, the interaction gates are different among different matrix elements, so we return a 2D array of MPOs here.
    def get_int_gates(self, dt):
        
        int_gates = np.empty((self.nsites, self.nsites), dtype=object)
        for m in range(self.nsites):
            for n in range(self.nsites):
                
                # Construct the MPO for matrix element (m,n)
                temporary_ops = []
                for i in range(self.nosc):
                    temporary_op = self.coups[m][i] * np.kron((self.a + self.a_dag),np.eye(self.localDim, dtype=complex)) - self.coups[n][i] * np.kron(np.eye(self.localDim, dtype=complex), (self.a + self.a_dag).T)
                    temporary_ops.append(scipy.linalg.expm(-1j * dt * temporary_op))
                
                # Combine local operators into an MPO, with bond dimensions of 1
                int_gates[m][n] = qtn.MPO_product_operator(temporary_ops)
                
        return int_gates
    
    # The electronic evolution operator is simply a matrix of complex numbers
    def get_U_els(self, dt):
        
        return scipy.linalg.expm(-1j * dt * self.el_ham)