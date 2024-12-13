import numpy as np

def quaternion_to_rotation_matrix(q):
    # 提取四元数的元素
    w, x, y, z = q
    
    # 计算旋转矩阵
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R

# 四元数
q = [-0.78489, 0.4981, 0.04358, -0.366]

# 转换为旋转矩阵
rotation_matrix = quaternion_to_rotation_matrix(q)
print(rotation_matrix*[0,0,0.099])
