import numpy as np
import quimb as qu
import quimb.tensor as qtn

# ------------------------
# 1. 随机生成一个 4-site MPO
# ------------------------
L = 4
d = 2  # 物理维度

# 随机哈密顿量 -> 演化算符
H = qu.rand_herm(d**L)
U = qu.expm(-0.1j * H)

# 转换成 MPO
mpo = qtn.MatrixProductOperator.from_dense(U, dims=d, L=L, cutoff=1e-10)
print("Initial MPO:", mpo)

# ------------------------
# 2. 随机生成一个 subMPO (作用在 sites [1, 2])
# ------------------------
# 随机两体算符 (4x4 unitary)
U12 = qu.rand_uni(d**2)
U12 = U12.reshape([d, d, d, d])  # (s1', s2', s1, s2)

# 转换成 subMPO（只在 site 1 和 2 上有非平凡作用）
submpo = qtn.MatrixProductOperator.from_dense(
            U12,
            dims=[d, d],   # physical dims for the two sites
            sites=[0, 2],
            L=L
        )
print("SubMPO on sites [1,2]:", submpo)

# ------------------------
# 3. 用 tensor_network_apply_op_op 作用 subMPO 在 MPO 上
# ------------------------
# 组合算符：保持 MPO 结构并可选压缩
mpo2 = qtn.tensor_network_apply_op_op(
    submpo, mpo,
    which_A="lower", which_B="upper",
    contract=True, fuse_multibonds=True, compress=True,
    max_bond=64, cutoff=1e-10, method="svd"
)
print("MPO after applying SubMPO:", mpo2)
