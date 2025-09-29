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
# print("os.cpu_count():", multiprocessing.cpu_count())
# print("SLURM_CPUS_ON_NODE:", os.environ.get('SLURM_CPUS_ON_NODE'))
# print("SLURM_CPUS_PER_TASK:", os.environ.get('SLURM_CPUS_PER_TASK'))

def get_alloc_cpus():
    """
    Determine how many CPUs we should use:
    priority: actual cpuset (sched_getaffinity) > SLURM_CPUS_ON_NODE > SLURM_CPUS_PER_TASK > os.cpu_count()
    """
    try:
        aff = os.sched_getaffinity(0)
        n_aff = len(aff)
    except Exception:
        n_aff = None

    def parse_first_env(*keys):
        for k in keys:
            v = os.environ.get(k)
            if v:
                try:
                    return int(v.split(',')[0])
                except Exception:
                    try:
                        return int(v)
                    except Exception:
                        pass
        return None

    n_slurm_node = parse_first_env('SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE')
    n_slurm_task = parse_first_env('SLURM_CPUS_PER_TASK')

    candidates = [x for x in (n_aff, n_slurm_node, n_slurm_task, multiprocessing.cpu_count()) if x is not None]
    if not candidates:
        return 1
    return min(candidates)

def worker(osc_state_and_localDim): # <--- Accept localDim as well
    osc_state, current_localDim = osc_state_and_localDim # Unpack
    
    np.random.seed(os.getpid() + int(time.time()*1000) % 10000)
    
    additional_osc_output_dic = {
        "1": np.diag(np.arange(current_localDim)),
    }
    
    trial_state = Totalsys_Pure(
        nsites=Pure_QT_config.nsites,
        nosc=Pure_QT_config.nosc,
        localDim=current_localDim, # <--- Use passed localDim
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
    
    return trial_state.results

def worker_chunk(chunk_and_localDim): # <--- Accept localDim
    osc_states, current_localDim = chunk_and_localDim # Unpack
    phonon_sum = None
    for osc_state in osc_states:
        # Pass both the state and the current localDim to the worker
        res = worker((osc_state, current_localDim))
        phonon = res["additional_osc_output"]
        if phonon_sum is None:
            phonon_sum = np.array(phonon, copy=True)
        else:
            phonon_sum += phonon
    return phonon_sum

# --- Main logic is now in a function ---
def run_simulation(current_localDim):
    """
    Runs the entire DAMPF simulation for a given local dimension.
    """
    print(f"\n{'='*20} Starting Simulation for localDim = {current_localDim} {'='*20}")
    
    # Create initial states using the specified localDim
    osc_state_nparray = utils.create_osc_initial_states(Pure_QT_config.nosc, Pure_QT_config.Ntraj, current_localDim, Pure_QT_config.temps)
    osc_state_array = list(osc_state_nparray)

    # Preconstruct gates for the specified localDim
    print("Constructing gates...")
    total_gates = utils.construct_all_gates(
        nsites=Pure_QT_config.nsites,
        elham=Pure_QT_config.elham,
        nosc=Pure_QT_config.nosc,
        freqs=Pure_QT_config.freqs,
        coups=Pure_QT_config.coups,
        temps=Pure_QT_config.temps,
        damps=Pure_QT_config.damps,
        localDim=current_localDim, # <--- Use passed localDim
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
    # <--- Package chunks with the current_localDim for the workers
    chunks_with_localDim = [(chunk, current_localDim) for chunk in chunks]
    print(f"Created {len(chunks_with_localDim)} chunks; per_worker ~ {per_worker}")

    sum_phonon_number = None

    print("Creating multiprocessing pool...")
    t_setup = time.time()
    with multiprocessing.Pool(processes=n_workers, initializer=init_gates, initargs=(total_gates,)) as pool:
        print(f"Pool created in {time.time() - t_setup:.2f} seconds")
        t = time.time()
        print("Simulation started, timing...")

        # Map chunks to workers
        for phonon_number_array in pool.imap_unordered(worker_chunk, chunks_with_localDim):
            if sum_phonon_number is None:
                sum_phonon_number = np.array(phonon_number_array, copy=True)
            else:
                sum_phonon_number += phonon_number_array
            gc.collect()

    total_time = time.time() - t
    print(f"Total time consumed for simulation: {total_time:.2f} seconds")

    ave_phonon_number = sum_phonon_number / float(Ntraj)

    # Plotting
    Time = np.arange(0, Pure_QT_config.time, Pure_QT_config.timestep)
    Time /= 3.1415926*2  # convert to units of 2pi
    Time = Time[:ave_phonon_number.shape[1]]
    for i in range(len(ave_phonon_number)):
        plt.plot(Time, ave_phonon_number[i].real, label=f'localDim={current_localDim}, Oscillator {i+1}')


if __name__ == "__main__":
    # --- Define the list of localDim values to simulate ---
    localDim_values = [5, 10, 15]

    # --- Loop through the values and run the simulation for each ---
    for dim in localDim_values:
        run_simulation(dim)
        
    plt.grid(True)
    plt.xlabel('Time($\omega t / 2 \pi$)')
    plt.ylabel('$<a^{\dagger}a>$')
    plt.title(f'Phonon Dynamics')
    plt.legend()
    plt.show()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figure_filename = f"figure_{timestamp}.pdf"
    params_filename = f"parameters_{timestamp}.txt"
    
    plt.savefig(figure_filename)
    plt.close() # <--- Close the plot to free memory
    
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
    
    print("\nAll simulations completed.")