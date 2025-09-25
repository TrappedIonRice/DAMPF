'''
This example demonstrates running DAMPF simulation using the pure state evolution assisted by quantum trajectories (QT) method, which can be readily put into parallel computing.
'''


import numpy as np
import Pure_QT_config
from Totalsys_Class import Totalsys_Pure, init_gates
import matplotlib.pyplot as plt
import gc
import multiprocessing
import time
import utils
from datetime import datetime
import os


# Define the worker function for parallel processing
def worker(osc_state, E):
    
    np.random.seed(os.getpid()+int(time.time()*1000) % 10000)
    
    elham = Pure_QT_config.elham
    for i in range(Pure_QT_config.nsites):
        if i < Pure_QT_config.nsites // 2:
            elham[i][i] = E / 2
        else:
            elham[i][i] = - E / 2
        
    print("A new trajectory started")
    trial_state = Totalsys_Pure(
        nsites=Pure_QT_config.nsites,
        nosc=Pure_QT_config.nosc,
        localDim=Pure_QT_config.localDim,
        elham=elham,
        freqs=Pure_QT_config.freqs,
        coups=Pure_QT_config.coups,
        damps=Pure_QT_config.damps,
        temps=Pure_QT_config.temps,
        time=Pure_QT_config.time,
        timestep=Pure_QT_config.timestep,
        el_initial_state=Pure_QT_config.el_initial_state,
        osc_state=osc_state
    )
    trial_state.Time_Evolve_Pure_QT(Pure_QT_config.timestep, Pure_QT_config.time, Pure_QT_config.maxBondDim)
    print("A trajectory ended")
    return trial_state.population


if __name__ == "__main__":

    # Prepare an array of initial oscillator states for all trajectories
    # osc_state_array = np.array([np.array([0]*Pure_QT_config.nosc) for _ in range(Pure_QT_config.Ntraj)]) # Use this line for zero-temperature initial states
    osc_state_nparray = utils.create_osc_initial_states(Pure_QT_config.nosc, Pure_QT_config.Ntraj, Pure_QT_config.localDim, Pure_QT_config.temps)
    osc_state_array = list(osc_state_nparray)
    
    E_array = np.array([1, 2, 3])
    
    for E in E_array:
        
        elham = Pure_QT_config.elham
        for i in range(Pure_QT_config.nsites):
            if i < Pure_QT_config.nsites // 2:
                elham[i][i] = E / 2
            else:
                elham[i][i] = - E / 2
        
        print("Constructing gates...")
        # Unlike other examples, these evolution gates need to be pre-constructed so that they can be shared across multiple processes.
        total_gates = utils.construct_all_gates(
            nsites=Pure_QT_config.nsites,
            elham=elham,
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
        
        ave_population = np.zeros((Pure_QT_config.nsites, int(Pure_QT_config.time / Pure_QT_config.timestep)))
        E_list = np.array([E]*Pure_QT_config.Ntraj)
        
        # Start parallel processing using multiprocessing Pool
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=init_gates, 
            initargs=(total_gates,)
        ) as pool:
            print(f"Pool created in {time.time() - t} seconds")
            t = time.time()
            print("Simulation started, timing...")
            workerargs = zip(osc_state_array, E_list)
            for pop_arr in pool.starmap(worker, workerargs):
                ave_population += pop_arr / Pure_QT_config.Ntraj
                gc.collect()
                
        print(f"Total time consumed for simulation: {time.time() - t} seconds")
    
    
        # Plot the averaged population dynamics after time evolution
        Time = np.arange(0, Pure_QT_config.time, Pure_QT_config.timestep)
        Time = Time[:ave_population.shape[1]]  # Ensure Time array matches the population data length
        Time /= (3.1415926*2)
        # for site in range(Pure_QT_config.nsites):
        #     plt.plot(Time, ave_population[site], label=f'Site {site}')        
        # plt.plot(Time, np.sum(ave_population, axis=0), label='Total')
        plt.plot(Time, ave_population[0] + ave_population[1],
                label=fr"$\epsilon = {E}\,\omega$")
    
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Population Dynamics')
    plt.legend()
    # plt.show()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"figure_{timestamp}.pdf"
    plt.savefig(filename)