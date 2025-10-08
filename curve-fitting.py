'''
This is the file for fitting the spectral density curve with a sum of double-Lorentzian functions. We utilize a global optimization algorithm wrapped in the `fcmaes` package, which is a package exclusively designed for handling optimization problems, but is relatively easy to be installed on Linux systems.
By double-Lorentzian, we mean a function of the form:
    L(ω) = (λ² * γ) / ((ω - ω₀)² + γ²) - (λ² * γ) / ((ω + ω₀)² + γ²)
In this specific example, we are fitting the spectral density functions with the form:
    J(ω) = A * ω^s * ω_c^{1-s} * exp(-ω/ω_c)
for s = 0.5, 1.0, 1.5 respectively.
'''



import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
from fcmaes import retry
from fcmaes.optimizer import Cma_cpp 
from datetime import datetime
import Pure_QT_config as config

D = config.DELTA
omega_l, omega_r = 0.85 * D, 1.15 * D   # We only fit the curve in this range
N = 1000    # number of nodes for the integral
Q = 2       # number of double-Lorentzian functions to use in the fitting
A = 0.1     # pre-factor in the spectral density functions
omega_c = 10 * D

# Get the Gauss-Legendre nodes and weights for efficient calculation of numerical integration
xlg, w = leggauss(N)
omega_nodes = 0.5*(omega_r+omega_l) + 0.5*(omega_r-omega_l)*xlg
weight_nodes = 0.5*(omega_r-omega_l)*w

s_array = [0.5, 1.0, 1.5]

# Define the spectral density functions to be fitted with different s values
def J1(omega): return A * np.sqrt(omega * omega_c) * np.exp(-omega / omega_c)
def J2(omega): return A * omega * np.exp(-omega / omega_c)
def J3(omega): return A * omega * np.sqrt(omega / omega_c) * np.exp(-omega / omega_c)

# The current approximation function using a sum of double-Lorentzian functions
def J_approx(omega, freq, gamma, lam):
    return np.sum((lam**2) * gamma / ((omega - freq)**2 + gamma**2)) - np.sum((lam**2) * gamma / ((omega + freq)**2 + gamma**2))

# Create the objective function for curve fitting
def make_objective(J_func):
    def f_obj(x):
        x = np.asarray(x, dtype=float)
        freq, gamma, lam = x[:Q], x[Q:2*Q], x[2*Q:3*Q]
        diff_sq = 0.0
        for n in range(N):
            Jn = J_func(omega_nodes[n])
            Vn = J_approx(omega_nodes[n], freq, gamma, lam)
            diff_sq += weight_nodes[n] * (Jn - Vn)**2
        return float(diff_sq)
    return f_obj

J_array = [J1, J2, J3]
f_array = [make_objective(J1), make_objective(J2), make_objective(J3)]

# Set the bounds for the parameters to be optimized
bounds = []
omega_max = 3.0 * D
for _ in range(Q):
    bounds.append((omega_l, omega_r))      # freq
for _ in range(Q):
    bounds.append((1e-3, 5))               # gamma
for _ in range(Q):
    bounds.append((0.0, 5))                # lambda

optimizer = Cma_cpp(int(50000))

fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

for i in range(3):
    
    ax = axes[i]
    res = retry.minimize(f_array[i], bounds, optimizer=optimizer)
    print(f"Best x for s={s_array[i]}:", np.array(res.x))
    print(f"Best f(x) for s={s_array[i]}:", res.fun)

    freq_est = res.x[:Q]
    gamma_est = res.x[Q:2*Q]
    lam_est = res.x[2*Q:3*Q]

    omega_plot = np.linspace(0.0, omega_max, 1000)
    J_plot = J_array[i](omega_plot)
    V_plot = np.array([J_approx(om, freq_est, gamma_est, lam_est) for om in omega_plot])

    ax.plot(omega_plot, J_plot, label='J(ω) (Real)')
    ax.plot(omega_plot, V_plot, label='Approx (Sum of Lorentzians)')

    for j in range(Q):
        lorentz = (lam_est[j]**2 * gamma_est[j]) / ((omega_plot - freq_est[j])**2 + gamma_est[j]**2) - (lam_est[j]**2 * gamma_est[j]) / ((omega_plot + freq_est[j])**2 + gamma_est[j]**2)
        ax.plot(omega_plot, lorentz, '--', alpha=0.7, label=f'Lorentz j={j+1}')

    ax.set_ylabel('J(ω)')
    ax.set_title(f'Curve fitting for s = {s_array[i]}')
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel('ω')
plt.tight_layout()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"figure_{timestamp}.pdf"
plt.savefig(filename)