import omni.isaac.lab.sim as sim_utils
import os
current_dir = os.path.dirname(__file__)
# 拼接路径，使其指向当前目录下的 iiwa7.usd 文件
usd_path = os.path.join(current_dir, 'iiwa7.usd')
cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
cfg.func("/World/iiwa7", cfg, translation=(0.0, 0.0, 1.05))