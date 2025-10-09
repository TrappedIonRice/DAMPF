# Spin-Boson System Simulation Using Tensor Network Techniques

A Python project for pure state QT (Quantum Trajectory) simulations and density matrix evolution. This repository provides several simulation examples and configuration files for different quantum system setups.

## Features
- Pure state QT simulation
- Density matrix evolution with fixed and adaptive steps
- Configurable system parameters
- Utility functions for analysis

## File Overview
- `Curve-Fitting.py`: Approximate the spectral density using a sum of double-Lorentzians, converting a continuous bath to a discrete but underdamped ones, thereby generating the parameters for next step simulation.
- `Main_Simulation_Example1(Pure_QT).py`: Pure state QT simulation
- `Main_Simulation_Example2(Rho_Fixed_Step).py`: Density matrix evolution with fixed step
- `Main_Simulation_Example3(Rho_Adaptive_Step).py`: Density matrix evolution with adaptive step
- `*_config.py`: Configuration files for each simulation type
- `Totalsys_Class.py`: Main system class
- `utils.py`: Utility functions
- `More Simulation Attempts Folder`: Various benchmarking attempts conducted during the development of this project, which include more abundant usages of this package.

## Installation
1. Clone the repository:
   ```powershell
   git clone <repo-url>
   ```
2. Install required Python packages:
   ```powershell
   pip install -r requirements.txt
   ```
   Don't worry if you find the kahypar package could not be installed, since it is only supported on Linux. This package is not required but recommended.
   The fcmaes package used for curve fitting is also recommended to be installed on Linux.

## Examples
For beginners, run one of the main simulation scripts depending on your needs:

- **Pure Quantum Trajectory Simulation**
  ```powershell
  python -u ".\Main_Simulation_Example1(Pure_QT).py"
  ```
- **Density Matrix Evolution (Fixed Step)**
  ```powershell
  python -u ".\Main_Simulation_Example2(Rho_Fixed_Step).py"
  ```
- **Density Matrix Evolution (Adaptive Step)**
  ```powershell
  python -u ".\Main_Simulation_Example3(Rho_Adaptive_Step).py"
  ```

## Hints
- Temperatures should be given in terms of nbar.
- When representing the total system in terms of density matrix (Example2 and Example3), the total number of modes can not be 1, in which case there exists an internal error of quimb package when doing MPS addition.
- To achieve better performance, consider deleting all the prompting print statements in the code.
- When using the QT Method, parallelization is highly recommended, but be aware that gates construction needs to be done separately as illustrated in the Example1.
- The default parameters are in the configuration files, which can be modified as needed. Also, if you want to vary some of the parameters to automatically generate multiple sets of data, you can modify the input parameters of class definitions or time_evolution functions in the main simulation files.



## Contact
Author: Boyi Zheng (University of Science and Technology of China, School of Gifted Young)

Email: byding0574@gmail.com
