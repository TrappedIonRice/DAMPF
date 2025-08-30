import numpy as np

# --------------------
# Basic system parameters
# --------------------
nsites = 2                 # number of sites
noscpersite = 1            # number of oscillators per site
nosc = noscpersite         # total number of oscillators
localDim = 3              # local dimension of oscillators
maxBondDim = 4             # maximal bond dimension of MPS
timestep = 0.5            # integration time-step in fs
time = 300               # total simulation time in fs

# --------------------
# Parameters for system dynamics (cm^-1)
# --------------------
# Site energies (Ω_n)

energies = [12430., 12405.]

# Exchange coupling (per site)
exchangepersite = 80.0

# Oscillators per site
freqspersite = [247]
tempKelvin = 0
huangRhysFactors = [0]
dampspersite = [0]
# huangRhysFactors = [0.056]
# dampspersite = [53]

# ================================================
# Conversion constants
# ================================================
_OmegaConv = 1.883651567e-4  # cm^-1 → fs^-1
_Tconv = 0.6950348           # K → cm^-1

# Convert energies
energies = _OmegaConv * np.array(energies)

# Exchange matrix (real & symmetric)
exchange = np.array([
    [0 if i == j else exchangepersite for i in range(nsites)]
    for j in range(nsites)
], dtype=float)
exchange *= _OmegaConv

# Frequencies of all oscillators (flattened)
freqs = _OmegaConv * np.array(freqspersite)

# Temperatures of oscillators in fs^-1
temps = np.full(nosc, _OmegaConv * _Tconv * tempKelvin)

# Coupling constants: [site_index, coupling_value...]
coups = []

for site in range(nsites):
    coup_values = [w * np.sqrt(s) for w, s in zip(freqs, huangRhysFactors)]
    coups.append(coup_values)

# Damping rates (flattened)
damps = _OmegaConv * np.array(dampspersite)


if __name__ == "__main__":
    print("energies =", energies)
    print("exchange matrix:\n", exchange)
    print("freqs =", freqs)
    print("temps =", temps)
    print("coups =", coups)
    print("damps =", damps)

