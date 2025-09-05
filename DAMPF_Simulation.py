'''
This is the main simulation script for the DAMPF method.
'''


import numpy as np
import DAMPF.config
from DAMPF.Totalsys_Class import Totalsys_Rho
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    TOTAL_RHO = Totalsys_Rho(DAMPF.config.nsites, DAMPF.config.noscpersite, DAMPF.config.nosc, DAMPF.config.localDim, DAMPF.config.temps, DAMPF.config.freqs, DAMPF.config.damps, DAMPF.config.coups, dt_array=DAMPF.config.dt_array)   # Initialize the total system density matrix in MPS form
    
    Time = TOTAL_RHO.Time_Evolve(total_time=DAMPF.config.time, initial_dt=DAMPF.config.initial_dt, max_bond_dim=DAMPF.config.maxBondDim, err_tol=DAMPF.config.error_tolerance, S1=DAMPF.config.S1, S2=DAMPF.config.S2)   # Time evolve the total system density matrix

    # Plot the population dynamics after time evolution
    # Time = np.arange(0, DAMPF.config.time, DAMPF.config.timestep)
    plt.plot(Time, np.array(TOTAL_RHO.populations).sum(axis=0), label='Total_trace')
    for i in range(DAMPF.config.nsites):
        plt.plot(Time, TOTAL_RHO.populations[i], label=f'Site {i+1}')
        
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('Population Dynamics')
    plt.grid('True')
    plt.show()