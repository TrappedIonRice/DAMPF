'''
This is to benchmarking results from Diego's paper regarding the average phonon number evolution (Fig 8b in the Appendix C).
'''



import os

# --- Environment variable setup (no changes) ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import Pure_QT_config
from Totalsys_Class import Totalsys_Pure, init_gates
import matplotlib.pyplot as plt
import gc
import multiprocessing
import time
import utils
from datetime import datetime
import math

# --- Helper functions (no changes) ---
def get_alloc_cpus():
    try:
        aff = os.sched_getaffinity(0)
        n_aff = len(aff)
    except Exception:
        n_aff = None
    def parse_first_env(*keys):
        for k in keys:
            v = os.environ.get(k)
            if v:
                try: return int(v.split(',')[0])
                except Exception:
                    try: return int(v)
                    except Exception: pass
        return None
    n_slurm_node = parse_first_env('SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE')
    n_slurm_task = parse_first_env('SLURM_CPUS_PER_TASK')
    candidates = [x for x in (n_aff, n_slurm_node, n_slurm_task, multiprocessing.cpu_count()) if x is not None]
    if not candidates: return 1
    return min(candidates)

# --- CORRECTED WORKER FUNCTIONS ---

def worker(osc_state_and_localDim):
    osc_state, current_localDim = osc_state_and_localDim
    np.random.seed(os.getpid() + int(time.time()*1000) % 10000)
    
    additional_osc_output_dic = {
        "1": np.diag(np.arange(current_localDim)),
    }
    
    trial_state = Totalsys_Pure(
        nsites=Pure_QT_config.nsites,
        nosc=Pure_QT_config.nosc,
        localDim=current_localDim,
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
        additional_osc_output_dic=additional_osc_output_dic,
    )
    trial_state.Time_Evolve_Pure_QT(Pure_QT_config.timestep, Pure_QT_config.time, Pure_QT_config.maxBondDim)
    
    # ✅ **CHANGE 1**: Return both results AND the max bond for this trajectory
    return (trial_state.results, trial_state.maxbond_throughout_whole_evolution)

def worker_chunk(chunk_and_localDim):
    osc_states, current_localDim = chunk_and_localDim
    phonon_sum = None
    max_bond_in_chunk = 1 # Initialize max bond for this chunk

    for osc_state in osc_states:
        # res is now a tuple: (results_dict, max_bond_for_traj)
        res, max_bond_for_traj = worker((osc_state, current_localDim))
        
        # Update max bond if a larger one is found in this trajectory
        if max_bond_for_traj > max_bond_in_chunk:
            max_bond_in_chunk = max_bond_for_traj
            
        phonon = res["additional_osc_output"]
        if phonon_sum is None:
            phonon_sum = np.array(phonon, copy=True)
        else:
            phonon_sum += phonon
            
    # ✅ **CHANGE 2**: Return both the sum AND the max bond found in this chunk
    return (phonon_sum, max_bond_in_chunk)

# --- Main logic function ---
def run_simulation(current_localDim):
    """
    Runs simulation and returns the max bond dimension found during this run.
    """
    print(f"\n{'='*20} Starting Simulation for localDim = {current_localDim} {'='*20}")
    
    osc_state_nparray = utils.create_thermal_osc_initial_states(Pure_QT_config.nosc, Pure_QT_config.Ntraj, current_localDim, Pure_QT_config.temps)
    osc_state_array = list(osc_state_nparray)

    print("Constructing gates...")
    total_gates = utils.construct_all_gates(
        nsites=Pure_QT_config.nsites, elham=Pure_QT_config.elham,
        nosc=Pure_QT_config.nosc, freqs=Pure_QT_config.freqs,
        coups=Pure_QT_config.coups, temps=Pure_QT_config.temps,
        damps=Pure_QT_config.damps, localDim=current_localDim,
        dt=Pure_QT_config.timestep
    )
    print("Gate construction finished.")

    Ntraj = Pure_QT_config.Ntraj
    n_alloc = get_alloc_cpus()
    n_workers = max(1, min(Ntraj, n_alloc))
    print(f"Determined: n_alloc={n_alloc}, using n_workers={n_workers}")

    per_worker = math.ceil(Ntraj / n_workers)
    chunks = [osc_state_array[i*per_worker: min((i+1)*per_worker, Ntraj)] for i in range(n_workers)]
    chunks = [c for c in chunks if len(c) > 0]
    chunks_with_localDim = [(chunk, current_localDim) for chunk in chunks]
    print(f"Created {len(chunks_with_localDim)} chunks; per_worker ~ {per_worker}")

    sum_phonon_number = None
    # ✅ **CHANGE 3**: Local variable to track max bond for THIS simulation run
    max_bond_this_run = 1

    print("Creating multiprocessing pool...")
    t_setup = time.time()
    with multiprocessing.Pool(processes=n_workers, initializer=init_gates, initargs=(total_gates,)) as pool:
        print(f"Pool created in {time.time() - t_setup:.2f} seconds")
        t = time.time()
        print("Simulation started, timing...")

        # Unpack the tuple returned by worker_chunk
        for phonon_number_array, max_bond_from_chunk in pool.imap_unordered(worker_chunk, chunks_with_localDim):
            if max_bond_from_chunk > max_bond_this_run:
                max_bond_this_run = max_bond_from_chunk

            if sum_phonon_number is None:
                sum_phonon_number = np.array(phonon_number_array, copy=True)
            else:
                sum_phonon_number += phonon_number_array
            gc.collect()

    total_time = time.time() - t
    print(f"Total time consumed for simulation: {total_time:.2f} seconds")

    ave_phonon_number = sum_phonon_number / float(Ntraj)

    Time = np.arange(0, Pure_QT_config.time, Pure_QT_config.timestep)
    Time /= 3.1415926*2
    Time = Time[:ave_phonon_number.shape[1]]
    for i in range(len(ave_phonon_number)):
        plt.plot(Time, ave_phonon_number[i].real, label=f'localDim={current_localDim}, Oscillator {i+1}')

    # ✅ **CHANGE 4**: Return the max bond found in this specific simulation
    return max_bond_this_run

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # This variable is now correctly managed in the main process only
    max_bond_dimension_throughout_all_simulations = 1
    
    localDim_values = [5, 10, 15]
    
    # Create the plot figure once, before the loop
    plt.figure(figsize=(10, 6))

    for dim in localDim_values:
        # ✅ **CHANGE 5**: Capture the returned max bond from the simulation run
        max_bond_from_run = run_simulation(dim)
        if max_bond_from_run > max_bond_dimension_throughout_all_simulations:
            max_bond_dimension_throughout_all_simulations = max_bond_from_run
            
    print(f"\nMaximum bond dimension encountered during all simulations: {max_bond_dimension_throughout_all_simulations}")
    
    # Finalize and save the single plot containing all lines
    plt.grid(True)
    plt.xlabel('Time($\\omega t / 2 \\pi$)')
    plt.ylabel('$<a^{\\dagger}a>$')
    plt.title('Phonon Dynamics for Different localDim')
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figure_filename = f"figure_{timestamp}.pdf"
    params_filename = f"parameters_{timestamp}.txt"
    
    # Save the figure before showing it
    plt.savefig(figure_filename)
    print(f"Figure saved to {figure_filename}")
    plt.show()
    plt.close()
    
    with open(params_filename, "w", encoding="utf-8") as f:
        
        f.write(f"Ntraj = {Pure_QT_config.Ntraj}\n")
        f.write(f"nsites = {Pure_QT_config.nsites}\n")
        f.write(f"nosc = {Pure_QT_config.nosc}\n")
        f.write(f"localDim_array = {localDim_values}\n")
        f.write(f"maxBondDim = {Pure_QT_config.maxBondDim}\n")
        f.write(f"timestep = {Pure_QT_config.timestep}\n")
        f.write(f"time = {Pure_QT_config.time}\n\n")

        f.write("el_initial_state =\n")
        f.write(f"{Pure_QT_config.el_initial_state}\n\n")

        f.write("elham =\n")
        f.write(f"{Pure_QT_config.elham}\n")

        f.write(f"freqs = {Pure_QT_config.freqs}\n")
        f.write(f"temps = {Pure_QT_config.temps}\n")

        f.write("coups =\n")
        f.write(f"{Pure_QT_config.coups}\n")

        f.write("damps =\n")
        f.write(f"{Pure_QT_config.damps}\n\n")

        f.write("additional_osc_jump_op_dic =\n")
        f.write(f"{Pure_QT_config.additional_osc_jump_op_dic}\n\n")

        f.write("additional_osc_output_dic =\n")
        f.write(f"{Pure_QT_config.additional_osc_output_dic}\n")
        
        f.write(f"\nMaximum bond dimension encountered during all simulations: {max_bond_dimension_throughout_all_simulations}\n")
    
    print("\nAll simulations completed.")