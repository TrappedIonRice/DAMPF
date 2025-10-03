import numpy as np
import Pure_QT_config
import Rho_Fixed_Step_config
from Totalsys_Class import Totalsys_Pure, init_gates
from Totalsys_Class import Totalsys_Rho_Fixed_Step
import matplotlib.pyplot as plt
import gc
import multiprocessing
import time
import utils
import os


# Define the worker function for parallel processing
def worker(osc_state):
    
    np.random.seed(os.getpid()+int(time.time()*1000) % 10000)
    
    print("A new trajectory started")
    trial_state = Totalsys_Pure(
        nsites=Pure_QT_config.nsites,
        nosc=Pure_QT_config.nosc,
        localDim=Pure_QT_config.localDim,
        elham=Pure_QT_config.elham,
        freqs=Pure_QT_config.freqs,
        coups=Pure_QT_config.coups,
        damps=Pure_QT_config.damps,
        temps=Pure_QT_config.temps,
        time=Pure_QT_config.time,
        timestep=Pure_QT_config.timestep,
        el_initial_state=Pure_QT_config.el_initial_state,
        osc_state=osc_state,
        additional_osc_jump_op_dic=Pure_QT_config.additional_osc_jump_op_dic,
        additional_osc_output_dic=Pure_QT_config.additional_osc_output_dic,
    )
    trial_state.Time_Evolve_Pure_QT(Pure_QT_config.timestep, Pure_QT_config.time, Pure_QT_config.maxBondDim)
    print("A trajectory ended")
    return trial_state.results


if __name__ == "__main__":

    # Prepare an array of initial oscillator states for all trajectories
    # osc_state_array = np.array([np.array([0]*Pure_QT_config.nosc) for _ in range(Pure_QT_config.Ntraj)]) # Use this line for zero-temperature initial states
    osc_state_nparray = utils.create_thermal_osc_initial_states(Pure_QT_config.nosc, Pure_QT_config.Ntraj, Pure_QT_config.localDim, Pure_QT_config.temps)
    osc_state_array = list(osc_state_nparray)
    
    ave_reduced_density_matrix = np.zeros((Pure_QT_config.nsites, Pure_QT_config.nsites, int(Pure_QT_config.time / Pure_QT_config.timestep)), dtype=complex)
    
    print("Constructing gates...")
    # Unlike other examples, these evolution gates need to be pre-constructed so that they can be shared across multiple processes.
    total_gates = utils.construct_all_gates(
        nsites=Pure_QT_config.nsites,
        elham=Pure_QT_config.elham,
        nosc=Pure_QT_config.nosc,
        freqs=Pure_QT_config.freqs,
        coups=Pure_QT_config.coups,
        temps=Pure_QT_config.temps,
        damps=Pure_QT_config.damps,
        localDim=Pure_QT_config.localDim,
        dt=Pure_QT_config.timestep
    )
    print("Gate construction finished.")
    
    print("Creating multiprocessing pool...")
    t = time.time()
    print("cpu_count =", multiprocessing.cpu_count())

    # Start parallel processing using multiprocessing Pool
    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(),
        initializer=init_gates, 
        initargs=(total_gates,)
    ) as pool:
        print(f"Pool created in {time.time() - t} seconds")
        t = time.time()
        print("Simulation started, timing...")
        for result in pool.imap(worker, osc_state_array):
            ave_reduced_density_matrix += result["reduced_density_matrix"] / Pure_QT_config.Ntraj
            gc.collect()
            
    print(f"Total time consumed for simulation: {time.time() - t} seconds")
    
    
    # Plot the averaged population dynamics after time evolution
    Time = np.arange(0, Pure_QT_config.time, Pure_QT_config.timestep)
    for site in range(Pure_QT_config.nsites):
        plt.plot(Time, ave_reduced_density_matrix[site][site].real, label=f'Site {site}')
    plt.plot(Time, np.trace(ave_reduced_density_matrix, axis1=0, axis2=1).real, label='Total')
    
    
    
    
    
    
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
        el_initial_state=Rho_Fixed_Step_config.el_initial_state
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
    
    
    
    
    
    
    
    
    
    
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Population Dynamics')
    plt.legend()
    plt.show()