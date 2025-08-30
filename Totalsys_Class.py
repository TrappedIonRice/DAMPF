'''
This module defines the Totalsys_Rho class, which is the core part of this whole project.
'''


import numpy as np
import utils
import quimb.tensor as qtn
import config
from tqdm import tqdm
import scipy.linalg


a = utils.annihilation_operator(config.localDim)
a_dag = a.conj().T


'''
The class Totalsys_Rho consists of information of the total system, including:

1. Parameters of the system (nsites, noscpersite, localDim, nosc, temps, freqs, damps, coups, energies, exchange, a, a_dag)
2. The total system density matrix in MPS form (rho), which is a 2D array of shape (nsites, nsites), with each element being an MPS (flattened MPO) representing the density matrices of the oscillators
3. The population dynamics during time evolution (populations), which will be updated after each time step by tracing out the diagonal MPS elements, resulting in nsites numbers,

as well as methods for time evolution, population update, and construction of various evolution gates:

1. __init__: Initializes everything
2. Time_Evolve: Time evolve the total system density matrix with the help of various gates
3. update_populations: Update the population dynamics after each time step
4. Various gates (as illustrated below)

'''
class Totalsys_Rho:
    
    def __init__(self, nsites, noscpersite, nosc, localDim, temps, freqs, damps, coups):
        
        thermal_mps = utils.create_thermal_mps(nosc, localDim, temps, freqs)
        
        # The zero_mps should be in the same structure as the thermal_mps, otherwise it will cause an error when adding two different-structured MPSs.
        zero_mps = thermal_mps.copy()
        for t in zero_mps.tensors:
            t.modify(data=t.data*0)
        
        # All of the elements in the intial rho is zero_mps, except the top-left corner
        self.rho = [[thermal_mps if (i==0) and (j==0) else zero_mps for i in range(nsites)] for j in range(nsites)]
        
        # Population initialization
        self.populations = np.zeros((nsites, int(config.time/config.timestep)))
        # self.test_populations = np.zeros((nsites, int(config.time/config.timestep)))
        
        # Parameter information initialization
        self.nsites = nsites
        self.noscpersite = noscpersite
        self.localDim = localDim
        self.nosc = nosc
        self.temps = temps
        self.freqs = freqs
        self.damps = damps
        self.coups = coups
        self.energies = config.energies
        self.exchange = config.exchange
        self.a = a
        self.a_dag = a_dag
    
    # def Test_Time_Evolve(self, timesteps, dt):
        
    #     rho = np.zeros((self.nsites, self.nsites), dtype=complex)
    #     rho[0][0] = 1.0 + 0j
        
    #     for step in tqdm(range(timesteps)):
            
    #         new_rho = np.empty((self.nsites, self.nsites), dtype=complex)
    #         for m in range(self.nsites):
    #             for n in range(self.nsites):
    #                 for k in range(self.nsites):
    #                     for l in range(self.nsites):
    #                         if (k == 0) and (l == 0):
    #                             new_rho[m][n] = self.get_el_coeffients(dt)[m][n][k][l] * rho[k][l]
    #                         else:
    #                             new_rho[m][n] += self.get_el_coeffients(dt)[m][n][k][l] * rho[k][l]
                                                                
    #         rho = new_rho   
                                                    
    #         for i in range(self.nsites):
    #             self.test_populations[i][step] = rho[i][i].real
        
    def Time_Evolve(self, timesteps, dt, max_bond_dim):
        
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
        
        # Three evolution gates
        osc_gates = self.get_osc_gates(dt)
        int_gates = self.get_int_gates(dt)
        el_coeffients = self.get_el_coeffients(dt)
        
        # Main time evolution loop
        for step in tqdm(range(timesteps)):
            
            # Electronic part evolution
            new_rho = np.empty((self.nsites, self.nsites), dtype=object)
            for m in range(self.nsites):
                for n in range(self.nsites):
                    for k in range(self.nsites):
                        for l in range(self.nsites):
                            if (k == 0) and (l == 0):
                                new_rho[m][n] = el_coeffients[m][n][k][l] * self.rho[k][l].copy()
                            else:
                                new_rho[m][n] += el_coeffients[m][n][k][l] * self.rho[k][l].copy()
                                                                
            self.rho = new_rho.copy()
            
            # Interaction part and oscillator part evolution
            for i in range(self.nsites):
                for j in range(self.nsites):
                    if i <= j:
                        self.rho[i][j] = self.rho[i][j].gate_with_mpo(int_gates[i][j])
                        self.rho[i][j] = self.rho[i][j].gate_with_mpo(osc_gates)
                        pass
                    else:
                        # Use the Hermitian property of the density matrix to reduce computation
                        self.rho[i][j] = self.rho[j][i].conj()
            
            # Compress the MPS after each time step to control the bond dimension
            for i in range(self.nsites):
                for j in range(self.nsites):
                    self.rho[i][j].compress(max_bond=max_bond_dim)
            
            # Update populations after each time step
            self.update_populations(step)
            
    def update_populations(self, step):
        
        for n in range(self.nsites):
            self.populations[n][step] = utils.trace_MPS(self.rho[n][n], self.nosc, self.nsites, self.localDim).real
    
    # The evolution gates of local oscillators are identical among all matrix elements (matrix elements refer to the elements in the total system density matrix), so we only return one MPO here.
    def get_osc_gates(self, dt):
        
        local_ops = []
        for i in range(self.nosc):
            local_ops.append(scipy.linalg.expm(-1j * dt * utils.local_ham_osc(self.freqs[i], self.localDim) + 2 * dt * self.damps[i] * utils.local_dissipator(self.freqs[i], self.temps[i], self.localDim)))
        
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
    
    # The electronic evolution are described by a set of coefficients, which are used to linearly combine the matrix elements to get the new matrix elements after time evolution.
    def get_el_coeffients(self, dt):
        
        cutoff=1e-12
        el_ham = self.exchange + np.diag(self.energies)
        
        U_el = scipy.linalg.expm(-1j * dt * el_ham)
        U_el_dag = U_el.conj().T
        
        Coefficients = np.zeros((self.nsites, self.nsites,self.nsites, self.nsites), dtype=complex)
        for m in range(self.nsites):
            for n in range(self.nsites):
                for k in range(self.nsites):
                    for l in range(self.nsites):
                        if np.abs(U_el[m][k] * U_el_dag[l][n]) > cutoff:
                            Coefficients[m][n][k][l] = U_el[m][k] * U_el_dag[l][n]
                            
        return Coefficients