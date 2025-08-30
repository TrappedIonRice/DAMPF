import numpy as np
import quimb.tensor as qtn

L = 1          # 站点数
phys_dim = 3   # 每个站点物理维度

# 生成两个不归一化的随机 product-state MPS
# MPS_product_state 默认是归一化的 product state，我们自己随机生成单站点向量
vecs1 = [np.random.randn(phys_dim) for _ in range(L)]
vecs2 = [np.random.randn(phys_dim) for _ in range(L)]

mps1 = qtn.MPS_product_state(vecs1)  # shape='lrp' -> (left, phys, right)
mps2 = qtn.MPS_product_state(vecs2)

# 计算和（quimb 对 MPS 实现了 __add__，会返回新的 MPS）
mps_sum = mps1 + mps2

print()
print(mps1)
print(mps2)
print(mps_sum)
print()
for i in range(L):
    print(mps1[i].data)
print()
for i in range(L):
    print(mps2[i].data)
print()
for i in range(L):
    print(mps_sum[i].data)