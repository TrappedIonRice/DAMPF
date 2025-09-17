'''
This example demonstrates running DAMPF simulation using the adaptive step time evolution of the total system density matrix.
'''


import numpy as np
import Rho_Adaptive_Step_config
from Totalsys_Class import Totalsys_Rho_Adaptive_Step
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    Total_Rho = Totalsys_Rho_Adaptive_Step(
        nsites=Rho_Adaptive_Step_config.nsites,
        nosc=Rho_Adaptive_Step_config.nosc,
        localDim=Rho_Adaptive_Step_config.localDim,
        elham=Rho_Adaptive_Step_config.elham,
        temps=Rho_Adaptive_Step_config.temps,
        freqs=Rho_Adaptive_Step_config.freqs,
        damps=Rho_Adaptive_Step_config.damps,
        coups=Rho_Adaptive_Step_config.coups,
        dt_array=Rho_Adaptive_Step_config.dt_array
    )   # Initialize the total system density matrix in MPS form
    
    Time = Total_Rho.Time_Evolve_Rho_Adaptive_Step(
        total_time=Rho_Adaptive_Step_config.time,
        initial_dt=Rho_Adaptive_Step_config.initial_dt,
        max_bond_dim=Rho_Adaptive_Step_config.maxBondDim,
        err_tol=Rho_Adaptive_Step_config.error_tolerance,
        S1=Rho_Adaptive_Step_config.S1,
        S2=Rho_Adaptive_Step_config.S2
    )   # Time evolve the total system density matrix


    # Plot the population dynamics after time evolution
    plt.plot(Time, np.array(Total_Rho.populations).sum(axis=0), label='Total_trace')
    for i in range(Rho_Adaptive_Step_config.nsites):
        plt.plot(Time, Total_Rho.populations[i], label=f'Site {i+1}')
        
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('Population Dynamics')
    plt.grid('True')
    plt.show()