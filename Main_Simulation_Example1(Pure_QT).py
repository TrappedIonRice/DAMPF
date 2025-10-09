'''
This example demonstrates running DAMPF simulation using the pure state evolution assisted by quantum trajectories (QT) method, which can be readily put into parallel computing.
'''

import os

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

# debug prints
# print("os.cpu_count():", multiprocessing.cpu_count())
# print("SLURM_CPUS_ON_NODE:", os.environ.get('SLURM_CPUS_ON_NODE'))
# print("SLURM_CPUS_PER_TASK:", os.environ.get('SLURM_CPUS_PER_TASK'))

def get_alloc_cpus():
    """
    Determine how many CPUs we should use:
    priority: actual cpuset (sched_getaffinity) > SLURM_CPUS_ON_NODE > SLURM_CPUS_PER_TASK > os.cpu_count()
    """
    # try sched_getaffinity (actual CPU set visible to this process)
    try:
        aff = os.sched_getaffinity(0)
        n_aff = len(aff)
    except Exception:
        n_aff = None

    # parse env vars
    def parse_first_env(*keys):
        for k in keys:
            v = os.environ.get(k)
            if v:
                try:
                    return int(v.split(',')[0])  # some clusters may give lists; try simple parse
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
    # choose the smallest candidate to avoid oversubscribe
    return min(candidates)

# worker runs a single trajectory and returns trial results
def worker(osc_state):
    
    # each process must have its own random seed, otherwise all trajectories done by multiprocessing will be identical
    np.random.seed(os.getpid() + int(time.time()*1000) % 10000)
    
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
    
    print("A trajectory finished")
    
    return (trial_state.results, trial_state.maxbond_throughout_whole_evolution)

# new: worker that accepts a chunk (list) of osc_states and returns the *sum* of reduced_density_matrix over that chunk
def worker_chunk(osc_states):
    
    partial_sum = None
    max_bond_in_chunk = 1 

    for osc_state in osc_states:

        res, max_bond_for_traj = worker(osc_state)
        
        if max_bond_for_traj > max_bond_in_chunk:
            max_bond_in_chunk = max_bond_for_traj

        rdm = res["reduced_density_matrix"]
        if partial_sum is None:
            partial_sum = np.array(rdm, copy=True)   # ensure we have a mutable copy
        else:
            partial_sum += rdm
            
    return (partial_sum, max_bond_in_chunk)

if __name__ == "__main__":
    # create initial states
    osc_state_nparray = utils.create_thermal_osc_initial_states(Pure_QT_config.nosc, Pure_QT_config.Ntraj, Pure_QT_config.localDim, Pure_QT_config.temps)
    # osc_state_nparray = np.zeros((1, 1, Pure_QT_config.localDim),dtype=complex)
    # osc_state_nparray[0][0][7] = 1.0 + 0.0j
    osc_state_array = list(osc_state_nparray)
    # print("osc_state_array:\n", osc_state_array)

    # preconstruct gates (same as you did) - these will be passed to initializer
    print("Constructing gates...")
    total_gates = utils.construct_all_gates(
        nsites=Pure_QT_config.nsites,
        elham=Pure_QT_config.elham,
        nosc=Pure_QT_config.nosc,
        freqs=Pure_QT_config.freqs,
        coups=Pure_QT_config.coups,
        temps=Pure_QT_config.temps,
        damps=Pure_QT_config.damps,
        localDim=Pure_QT_config.localDim,
        dt=Pure_QT_config.timestep,
        additional_osc_jump_op_dic=Pure_QT_config.additional_osc_jump_op_dic
    )
    print("Gate construction finished.")

    # Determine worker count safely
    Ntraj = Pure_QT_config.Ntraj
    n_alloc = get_alloc_cpus()
    n_workers = max(1, min(Ntraj, n_alloc))
    print(f"Determined: n_alloc={n_alloc}, using n_workers={n_workers}")

    # Build chunks: each worker gets ~Ntraj/n_workers trajectories
    per_worker = math.ceil(Ntraj / n_workers)
    chunks = [osc_state_array[i*per_worker: min((i+1)*per_worker, Ntraj)] for i in range(n_workers)]
    # remove empty chunks (in case Ntraj < n_workers)
    chunks = [c for c in chunks if len(c) > 0]
    print(f"Created {len(chunks)} chunks; per_worker ~ {per_worker}")

    # prepare accumulator for sums
    sum_reduced_density_matrix = None
    max_bond_dimension_throughout_simulation = 1

    print("Creating multiprocessing pool...")
    t_setup = time.time()
    # IMPORTANT: pass initializer so each worker installs total_gates (keeps your pattern)
    # use processes=n_workers rather than cpu_count()
    with multiprocessing.Pool(processes=n_workers, initializer=init_gates, initargs=(total_gates,)) as pool:
        print(f"Pool created in {time.time() - t_setup:.2f} seconds")
        t = time.time()
        print("Simulation started, timing...")

        for partial_rdm, max_bond_from_chunk in pool.imap_unordered(worker_chunk, chunks):
            
            if max_bond_from_chunk > max_bond_dimension_throughout_simulation:
                max_bond_dimension_throughout_simulation = max_bond_from_chunk

            if sum_reduced_density_matrix is None:
                sum_reduced_density_matrix = np.array(partial_rdm, copy=True)
            else:
                sum_reduced_density_matrix += partial_rdm
            gc.collect()

    total_time = time.time() - t
    print(f"Total time consumed for simulation (parallel part): {total_time:.2f} seconds")
    print(f"Maximum bond dimension encountered during the simulation: {max_bond_dimension_throughout_simulation}")


    # now average
    ave_reduced_density_matrix = sum_reduced_density_matrix / float(Ntraj)

    # Plot the averaged reduced density matrix dynamics after time evolution
    Time = np.arange(0, Pure_QT_config.time, Pure_QT_config.timestep)
    Time = Time[:ave_reduced_density_matrix.shape[2]]  # in case of slight size mismatch due to rounding
    # for site in range(Pure_QT_config.nsites):
    #     plt.plot(Time, ave_reduced_density_matrix[site][site].real, label=f'Site {site}')
    # plt.plot(Time, np.trace(ave_reduced_density_matrix, axis1=0, axis2=1).real, label='Total')
    plt.plot(Time, 2 * ave_reduced_density_matrix[0][1].real, label=f'real')
    plt.plot(Time, 2 * ave_reduced_density_matrix[0][1].imag, label=f'imag')

    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Coherence')
    plt.title('Coherence Dynamics')
    plt.legend()
    # plt.show()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"figure_{timestamp}.pdf"
    plt.savefig(filename)
    
    filename1 = f"parameters_{timestamp}.txt"
    
    with open(filename1, "w", encoding="utf-8") as f:
        
        f.write(f"Ntraj = {Pure_QT_config.Ntraj}\n")
        f.write(f"nsites = {Pure_QT_config.nsites}\n")
        f.write(f"nosc = {Pure_QT_config.nosc}\n")
        f.write(f"localDim = {Pure_QT_config.localDim}\n")
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
        
        f.write(f"\nTotal time consumed for simulation: {total_time:.2f} seconds\n")

        f.write(f"Maximum bond dimension encountered during the simulation: {max_bond_dimension_throughout_simulation}\n")