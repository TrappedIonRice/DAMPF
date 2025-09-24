
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmam, mcsolve, mesolve

gamma = 0.2
omega = 1.0
times = np.linspace(0, 10, 400)

sx = sigmax()
sm = sigmam()
H = 0.5 * omega * sx
rho0 = basis(2,0) * basis(2,0).dag()

c_ops = [np.sqrt(gamma) * sm]
e_ops = [sm.dag() * sm]

res = mesolve(H, rho0, times, c_ops, e_ops)

plt.plot(times, res.expect[0], label='⟨σ⁺σ⁻⟩ (density matrix)')

psi0 = basis(2, 0)
res = mcsolve(H, psi0, times, c_ops, e_ops, ntraj=1000)

# 绘图：期望值随时间
plt.plot(times, res.expect[0])
plt.xlabel("time")
plt.ylabel("Excitation probability ⟨σ⁺σ⁻⟩")
plt.title("Monte Carlo trajectories (averaged) via qutip.mcsolve")
plt.grid(True)
plt.show()
