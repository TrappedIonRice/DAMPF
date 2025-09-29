import quimb as qu
import quimb.tensor as qtn

# 系统大小和物理维数
N = 5       # sites
phys_dim = 2
bond_dim = 3

# 随机生成 MPS
mps = qtn.MPS_rand_state(N, phys_dim=phys_dim, bond_dim=bond_dim)

print(mps.max_bond())