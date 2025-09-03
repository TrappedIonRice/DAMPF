# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:45:06 2025

@author: zhumj
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:02:12 2024

@author: zhumj
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
#%%
def H_ET(E,V,omega1,omega2,g1,g2,cutoff):
    pI = qeye(cutoff); sI = qeye(2);
    xop = create(cutoff)+destroy(cutoff)
    term1 = tensor(0.5*E*sigmaz()+V*sigmax(),pI,pI)
    term2 = 0.5*g1*tensor(sigmaz(),xop,pI) + 0.5*g2*tensor(sigmaz(),pI,xop)
    term3 = omega1*tensor(sI,num(cutoff),pI) + omega2*tensor(sI,pI,num(cutoff))
    return (term1+term2+term3)
def Lind(gamma1,gamma2,nbar1,nbar2,cutoff):
    pI = qeye(cutoff); sI = qeye(2);
    clist = []
    cm1 = tensor(sI,destroy(cutoff),pI)
    cm2 = tensor(sI,pI,destroy(cutoff))
    #coeff1 = np.sqrt(2*np.pi*gamma)
    coeff1 = np.sqrt(gamma1);coeff2 = np.sqrt(gamma2)
    clist = [ coeff1*np.sqrt(1+nbar1)*cm1, coeff1* np.sqrt(nbar1)*cm1.dag(),
             coeff2*np.sqrt(1+nbar2)*cm2, coeff2* np.sqrt(nbar2)*cm2.dag()] 
    return clist
#%%
cutoff = 10
E = 1
omega1 = 1; omega2 = 1.2; V = 0.05; g1 = 1; g2 = 1.2;
gamma1 = 0.05; gamma2 = 0.05; n1 = 0; n2 = 0
H = H_ET(E,V,omega1,omega2,g1,g2,cutoff)

g_eff1 = -0.5*g1/omega1; g_eff2 = -0.5*g2/omega2
#diplaced initial state
ed_ket = tensor(basis(2,0),displace(cutoff,g_eff1)*fock(cutoff,0),displace(cutoff,g_eff2)*fock(cutoff,0))
#fock state 
e0_ket = tensor(basis(2,0),fock(cutoff,0),fock(cutoff,0))
clist1 =   Lind(gamma1,gamma2,n1,n2,cutoff)
rho0 = e0_ket*e0_ket.dag()
#rho0 = ed_ket*ed_ket.dag()
times = np.linspace(0,200,100000)
#%% evolve 
elist1 = [tensor(0.5*sigmaz()+0.5,qeye(cutoff),qeye(cutoff))]
#coherent (with displaced state)
result1 = sesolve(H,ed_ket,times,elist1,progress_bar=True,options=Options(nsteps=100000))
#dissipative
#result2 = mesolve(H,rho0,times,clist1,elist1,progress_bar=True,options=Options(nsteps=100000))
#%%

p1 = result1.expect[0]
#p2 = result2.expect[0]
#p2= expect(elist[1], result.states)
fig = plt.figure()
plt.clf()
plt.plot(times,p1,"-",label="coherent")
#plt.plot(times,p2,"-",label="dissipative")
plt.rcParams['figure.dpi']= 200
#title = r'$\delta_{tilt} = $' + str(-delta)+r', $\Delta E_z = $' + str(deltaE/delta)+r'$\delta_{tilt}$, $\Omega_x = $' + str(round(omegax/delta,2))+r', $\alpha = $' + str(round(g_fac/2,2))+r', $g = $'+ str(round(laser1.Omega_eff/delta,4))+r'$\delta_{tilt}$, $\bar{n}_0 = $'+str(nfock)
plt.xlabel(r'$t$',fontsize = 14)
plt.ylabel(r'$P_e$',fontsize = 14)
#lt.title(title)
plt.grid() 
#plt.xlim(0,max(tplot))
plt.ylim(0,1.05)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend()
plt.show()