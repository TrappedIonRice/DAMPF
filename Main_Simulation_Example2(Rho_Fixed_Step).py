'''
This example demonstrates running DAMPF simulation using the fixed step time evolution of the total system density matrix.
'''


import numpy as np
import Rho_Fixed_Step_config
from Totalsys_Class import Totalsys_Rho_Fixed_Step
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    Total_Rho = Totalsys_Rho_Fixed_Step(
        nsites=Rho_Fixed_Step_config.nsites,
        nosc=Rho_Fixed_Step_config.nosc,
        localDim=Rho_Fixed_Step_config.localDim,
        temps=Rho_Fixed_Step_config.temps,
        freqs=Rho_Fixed_Step_config.freqs,
        damps=Rho_Fixed_Step_config.damps,
        coups=Rho_Fixed_Step_config.coups,
        time=Rho_Fixed_Step_config.time,
        timestep=Rho_Fixed_Step_config.timestep,
        elham=Rho_Fixed_Step_config.elham
    )   # Initialize the total system density matrix in MPS form
    
    Total_Rho.Time_Evolve_Rho_Fixed_Step(
        time=Rho_Fixed_Step_config.time,
        dt=Rho_Fixed_Step_config.timestep,
        max_bond_dim=Rho_Fixed_Step_config.maxBondDim
    )   # Time evolve the total system density matrix


    # Plot the population dynamics after time evolution
    Time = np.arange(0, Rho_Fixed_Step_config.time, Rho_Fixed_Step_config.timestep)
    plt.plot(Time, Total_Rho.populations.sum(axis=0), label='Total_trace')
    for i in range(Rho_Fixed_Step_config.nsites):
        plt.plot(Time, Total_Rho.populations[i], label=f'Site {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('Population Dynamics')
    plt.grid('True')
    plt.show()