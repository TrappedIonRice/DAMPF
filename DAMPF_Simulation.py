'''
This is the main simulation script for the DAMPF method.
'''


import numpy as np
import config
from Totalsys_Class import Totalsys_Rho
import matplotlib.pyplot as plt


TOTAL_RHO = Totalsys_Rho(config.nsites, config.noscpersite, config.nosc, config.localDim, config.temps, config.freqs, config.damps, config.coups)   # Initialize the total system density matrix in MPS form

TOTAL_RHO.Time_Evolve(timesteps=int(config.time/config.timestep), dt=config.timestep, max_bond_dim=config.maxBondDim)   # Time evolve the total system density matrix

# Plot the population dynamics after time evolution
Time = np.arange(0, config.time, config.timestep)
plt.plot(Time, TOTAL_RHO.populations.sum(axis=0), label='Total_trace')
for i in range(config.nsites):
    plt.plot(Time, TOTAL_RHO.populations[i], label=f'Site {i+1}')
    
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Population Dynamics')
plt.grid('True')
plt.show()