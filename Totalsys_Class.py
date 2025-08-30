import numpy as np
import utils
import quimb.tensor as qtn
import config
from tqdm import tqdm
import scipy.linalg


a = utils.annihilation_operator(config.localDim)
a_dag = a.conj().T


class Totalsys_Rho:
    
    def __init__(self, nsites, noscpersite, nosc, localDim, temps, freqs, damps, coups):
        
        thermal_mps = utils.create_thermal_mps(nosc, localDim, temps, freqs)
        zero_mps = thermal_mps.copy()
        for t in zero_mps.tensors:
            t.modify(data=t.data*0)
        self.rho = [[thermal_mps if (i==0) and (j==0) else zero_mps for i in range(nsites)] for j in range(nsites)]
        
        self.populations = np.zeros((nsites, int(config.time/config.timestep)))
        self.test_populations = np.zeros((nsites, int(config.time/config.timestep)))
        
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
    
    def Test_Time_Evolve(self, timesteps, dt):
        
        rho = np.zeros((self.nsites, self.nsites), dtype=complex)
        rho[0][0] = 1.0 + 0j
        
        for step in tqdm(range(timesteps)):
            
            # if step == 10:
                
            #     print()
            #     for i in range(self.nsites):
            #         for j in range(self.nsites):
            #             print(rho[i][j])
            #     exit()
            
            new_rho = np.empty((self.nsites, self.nsites), dtype=complex)
            for m in range(self.nsites):
                for n in range(self.nsites):
                    for k in range(self.nsites):
                        for l in range(self.nsites):
                            if (k == 0) and (l == 0):
                                new_rho[m][n] = self.get_el_coeffients(dt)[m][n][k][l] * rho[k][l]
                            else:
                                new_rho[m][n] += self.get_el_coeffients(dt)[m][n][k][l] * rho[k][l]
                                                                
            rho = new_rho   
                                                    
            for i in range(self.nsites):
                self.test_populations[i][step] = rho[i][i].real
        
    def Time_Evolve(self, timesteps, dt, max_bond_dim):
        
        osc_gates = self.get_osc_gates(dt)
        int_gates = self.get_int_gates(dt)
        el_coeffients = self.get_el_coeffients(dt)
        
        # print()
        # print(el_coeffients)
        # print()
        
        for step in tqdm(range(timesteps)):
            
            # if step == 10:
                
            #     print()
            #     for i in range(self.nsites):
            #         for j in range(self.nsites):
            #             print(self.rho[i][j][0].data)
            #     exit()
            
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
                                                    
            for i in range(self.nsites):
                for j in range(self.nsites):
                    if i <= j:
                        self.rho[i][j] = self.rho[i][j].gate_with_mpo(int_gates[i][j])
                        self.rho[i][j] = self.rho[i][j].gate_with_mpo(osc_gates)
                        pass
                    else:
                        self.rho[i][j] = self.rho[j][i].conj()
                        
            for i in range(self.nsites):
                for j in range(self.nsites):
                    self.rho[i][j].compress(max_bond=max_bond_dim)
            
            self.update_populations(step)
            
    def update_populations(self, step):
        
        for n in range(self.nsites):
            self.populations[n][step] = utils.trace_MPS(self.rho[n][n], self.nosc, self.nsites, self.localDim).real
            
    def get_osc_gates(self, dt):
        
        local_ops = []
        for i in range(self.nosc):
            local_ops.append(scipy.linalg.expm(-1j * dt * utils.local_ham_osc(self.freqs[i], self.localDim) + 2 * dt * self.damps[i] * utils.local_dissipator(self.freqs[i], self.temps[i], self.localDim)))
            
        return qtn.MPO_product_operator(local_ops)
    
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
    
    def get_el_coeffients(self, dt):
        
        cutoff=1e-12
        el_ham = self.exchange + np.diag(self.energies)
        
        U_el = scipy.linalg.expm(-1j * dt * el_ham)
        U_el_dag = U_el.conj().T
        
        # print("U_el =\n", U_el)
        # print("U_el_dag =\n", U_el_dag)
        # exit()
        
        Coefficients = np.zeros((self.nsites, self.nsites,self.nsites, self.nsites), dtype=complex)
        for m in range(self.nsites):
            for n in range(self.nsites):
                for k in range(self.nsites):
                    for l in range(self.nsites):
                        if np.abs(U_el[m][k] * U_el_dag[l][n]) > cutoff:
                            Coefficients[m][n][k][l] = U_el[m][k] * U_el_dag[l][n]
                            
        return Coefficients                       