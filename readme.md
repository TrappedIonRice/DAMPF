<<<<<<< HEAD
For first running attempt, please type "python .\DAMPF_Simulation" in the terminal

Parameters can be modified in DAMPF\config.py

Dependency list: numpy scipy matplotlib tqdm quimb
=======
# DAMPF_Final

A Python project for quantum trajectory simulations and density matrix evolution. This repository provides several simulation examples and configuration files for different quantum system setups.

## Features
- Pure quantum trajectory simulation
- Density matrix evolution with fixed and adaptive steps
- Configurable system parameters
- Utility functions for analysis

## Installation
1. Clone the repository:
   ```powershell
   git clone <repo-url>
   ```
2. Install required Python packages:
   ```powershell
   pip install -r requirements.txt
   ```
   Note: The package kahypar is only supported on Linux, which is not required but recommended.

## Usage
Run one of the main simulation scripts depending on your needs:

- **Pure Quantum Trajectory Simulation**
  ```powershell
  python Main_Simulaiton_Example1(Pure_QT).py
  ```
- **Density Matrix Evolution (Fixed Step)**
  ```powershell
  python Main_Simulation_Example2(Rho_Fixed_Step).py
  ```
- **Density Matrix Evolution (Adaptive Step)**
  ```powershell
  python Main_Simulation_Example3(Rho_Adaptive_Step).py
  ```

Configuration files (e.g., `Pure_QT_config.py`, `Rho_Fixed_Step_config.py`) can be edited to change simulation parameters.

## File Overview
- `Main_Simulaiton_Example1(Pure_QT).py`: Pure quantum trajectory simulation
- `Main_Simulation_Example2(Rho_Fixed_Step).py`: Density matrix evolution with fixed step
- `Main_Simulation_Example3(Rho_Adaptive_Step).py`: Density matrix evolution with adaptive step
- `*_config.py`: Configuration files for each simulation type
- `Totalsys_Class.py`: Main system class
- `utils.py`: Utility functions

## Contact
Author: Boyi Zheng
Email: byding0574@gmail.com
>>>>>>> 94220de (Integrated Version)
