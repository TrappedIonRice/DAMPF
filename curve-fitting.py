import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from fcmaes import retry
from fcmaes.optimizer import Cma_cpp 
from datetime import datetime


omega_l, omega_r = 0.0, 300.0
N = 1000
Q = 6
A = 1.0
omega_c = 100.0
xlg, w = leggauss(N)
omega_nodes = 0.5*(omega_r+omega_l) + 0.5*(omega_r-omega_l)*xlg
weight_nodes = 0.5*(omega_r-omega_l)*w

def J(omega):
    return A * omega * np.exp(-omega / omega_c)

def V_tilde(omega, freq, gamma, lam):

    return np.sum((lam**2) * gamma / ((omega - freq)**2 + gamma**2))

def f_obj(x):
    x = np.asarray(x, dtype=float)
    freq = x[:Q]
    gamma = x[Q:2*Q]
    lam = x[2*Q:3*Q]
    # # 数值保护：强烈惩罚不合法 gamma 或 nan/inf
    # if np.any(~np.isfinite(x)):
    #     return 1e12
    # if np.any(gamma <= 0):
    #     return 1e10 + np.sum(np.where(gamma<=0, 1e6, 0))

    diff_sq = 0.0

    for n in range(N):
        Jn = J(omega_nodes[n])
        Vn = V_tilde(omega_nodes[n], freq, gamma, lam)
        diff_sq += weight_nodes[n] * (Jn - Vn)**2

    return float(diff_sq)

def f_obj_transformed(x):
    # x layout: [freq_0..freq_{Q-1}, log_gamma_0..log_gamma_{Q-1}, log_lam_0..log_lam_{Q-1}]
    x = np.asarray(x, dtype=float)
    freq = np.sort(x[:Q])                   # enforce ordering -> removes permutation symmetry
    log_gamma = x[Q:2*Q]
    log_lam = x[2*Q:3*Q]
    gamma = np.exp(log_gamma)
    lam = np.exp(log_lam)

    diff_sq = 0.0
    for n in range(N):
        Jn = J(omega_nodes[n])
        Vn = V_tilde(omega_nodes[n], freq, gamma, lam)
        diff_sq += weight_nodes[n] * (Jn - Vn)**2

    return float(diff_sq)


bounds = []
omega_max = 300.0
for _ in range(Q):
    bounds.append((0.0, omega_max))               # freq
for _ in range(Q):
    bounds.append((np.log(1e-6), np.log(50.0)))   # log_gamma
for _ in range(Q):
    bounds.append((np.log(1e-8), np.log(200.0)))  # log_lam


# 先用比较严格的预算和随机种子，便于重复实验
res = differential_evolution(f_obj_transformed, bounds, maxiter=50, popsize=15, seed=1234)
print("best fun:", res.fun)
print("best x:", res.x)

freq_est = np.exp(res.x[:Q])
gamma_est = np.exp(res.x[Q:2*Q])
lam_est = np.exp(res.x[2*Q:3*Q])

omega_plot = np.linspace(0.0, omega_max, 1000)
J_plot = J(omega_plot)
V_plot = np.array([V_tilde(om, freq_est, gamma_est, lam_est) for om in omega_plot])

plt.plot(omega_plot, J_plot, label='J(ω) (Real)')
plt.plot(omega_plot, V_plot, label='Approx (Sum of Lorentzians)')

lorentzians = np.vstack([
    (lam_est[j]**2 * gamma_est[j]) / ((omega_plot - freq_est[j])**2 + gamma_est[j]**2)
    for j in range(Q)
])

for j in range(Q):
    plt.plot(omega_plot, lorentzians[j], linestyle='--', alpha=0.7, label=f'Lorentz j={j+1}')

plt.grid(True)
plt.xlabel('ω')
plt.ylabel('Value')
plt.legend()
plt.title('Fitting result')

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"figure_{timestamp}.pdf"
plt.savefig(filename)