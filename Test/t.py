# 保存为 build_mpo_from_arrays.py 并在已安装 quimb 的环境中运行
import numpy as np
import quimb.tensor as qtn

def arrays_to_mpo(arrays, shape='lrud'):

    arrays = [np.asarray(a) for a in arrays]

    for i, a in enumerate(arrays):
        if a.ndim != 4:
            raise ValueError(f"arrays[{i}] must be 4D (left,right,up,down). Got shape {a.shape}")
        if i == 0:
            phys_up, phys_down = a.shape[2], a.shape[3]
        else:
            if a.shape[2] != phys_up or a.shape[3] != phys_down:
                raise ValueError(f"Physical dims mismatch at site {i}: expected (*,*,{phys_up},{phys_down}), got {a.shape}")
            
    mpo = qtn.MatrixProductOperator(arrays, shape=shape)
    return mpo


# --------------------------
# 示例：如何用给定 bond-dimension 生成数组列表并构造 MPO
# --------------------------
if __name__ == "__main__":
    L = 5           # 站点数
    phys_dim = 2    # 每站点物理维度
    bond_dim = 3    # 希望的内部 bond dimension

    # 生成示例数组：OBC（开边界），左右端 bond = 1
    arrays = []
    for site in range(L):
        left = 1 if site == 0 else bond_dim
        right = 1 if site == L - 1 else bond_dim
        # 复数随机张量示例（实际可替换为用户给定的数组）
        arr = (np.random.randn(left, right, phys_dim, phys_dim) +
               1j * np.random.randn(left, right, phys_dim, phys_dim))
        arrays.append(arr)

    mpo = arrays_to_mpo(arrays)
    print("MPO 构造成功：", mpo)

