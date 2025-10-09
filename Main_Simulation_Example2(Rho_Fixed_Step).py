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
        elham=Rho_Fixed_Step_config.elham,
        el_initial_state=Rho_Fixed_Step_config.el_initial_state,
        additional_osc_output_dic=Rho_Fixed_Step_config.additional_osc_output_dic
    )   # Initialize the total system density matrix in MPS form
    
    Total_Rho.Time_Evolve_Rho_Fixed_Step(
        time=Rho_Fixed_Step_config.time,
        dt=Rho_Fixed_Step_config.timestep,
        max_bond_dim=Rho_Fixed_Step_config.maxBondDim
    )   # Time evolve the total system density matrix


    # Plot the reduced density matrix dynamics after time evolution
    Time = np.arange(0, Rho_Fixed_Step_config.time, Rho_Fixed_Step_config.timestep)
    Time = Time[:Total_Rho.results["reduced_density_matrix"].shape[2]]  # in case of slight size mismatch due to rounding
    # plt.plot(Time, np.trace(Total_Rho.results["reduced_density_matrix"], axis1=0, axis2=1).real, label='Total_trace')
    # for i in range(Rho_Fixed_Step_config.nsites):
    #     plt.plot(Time, Total_Rho.results["reduced_density_matrix"][i][i], label=f'Site {i+1}')
    # plt.plot(Time, Total_Rho.results["reduced_density_matrix"][0][1].real, label='Re(rho_12)')
    # plt.plot(Time, Total_Rho.results["reduced_density_matrix"][0][1].imag, label='Im(rho_12)')
    plt.plot(Time, Total_Rho.results["additional_osc_output"][0].real, label='ave_phonon_osc1')
    plt.plot(Time, Total_Rho.results["additional_osc_output"][1].real, label='ave_phonon_osc2')
    
    plt.xlabel('Time')
    # plt.ylabel('Population')
    # plt.title('Population Dynamics')
    plt.ylabel('Average Phonon Number')
    plt.title('Phonon Dynamics')
    plt.legend()
    plt.grid('True')
    plt.show()