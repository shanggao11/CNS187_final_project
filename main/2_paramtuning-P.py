#%%
import numpy as np
import matplotlib.pyplot as plt
from funs import funs
import os

script_name="2_paramtuning-P"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
size=(8,8)
alpha=0.1
beta=0.02
gamma=0.02
num_batches=1500
settling_steps=300

configs_matched=[
    {"name":"p_line=1/8, p=1/8", "size":size, "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"p_line=1/4, p=1/4", "size":size, "p_line":1/4, "p":1/4, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"p_line=1/2, p=1/2", "size":size, "p_line":1/2, "p":1/2, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps},
]

configs_mismatch=[
    {"name":"p_line=1/8, p=1/8", "size":size, "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"p_line=1/4, p=1/8", "size":size, "p_line":1/4, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"p_line=1/2, p=1/8", "size":size, "p_line":1/2, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps},
]

configs_size=[
    {"name":"8x8, p_line=1/8, p=1/8", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"16x16, p_line=1/16, p=1/16", "size":(16,16), "p_line":1/16, "p":1/16, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":2500, "settling_steps":settling_steps},
    {"name":"16x16, p_line=1/8, p=1/8", "size":(16,16), "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":2500, "settling_steps":settling_steps},
]

#%%
matched=funs.run_configs(configs_matched)
mismatch=funs.run_configs(configs_mismatch)
size_results=funs.run_configs(configs_size)

#%%
funs.plot_summary(matched, "p_line and p both changed", os.path.join(save_dir, f"{script_name}_summary_matched.png"))
funs.plot_summary(mismatch, "p_line changed, p fixed at 1/8", os.path.join(save_dir, f"{script_name}_summary_mismatch.png"))
funs.plot_summary(size_results, "image size experiments", os.path.join(save_dir, f"{script_name}_summary_size.png"))

#%%
funs.plot_activity(matched, "activity: matched p", os.path.join(save_dir, f"{script_name}_activity_matched.png"))
funs.plot_activity(mismatch, "activity: mismatched p", os.path.join(save_dir, f"{script_name}_activity_mismatch.png"))
funs.plot_activity(size_results, "activity: image size", os.path.join(save_dir, f"{script_name}_activity_size.png"))

#%%
for i,r in enumerate(matched):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_matched_{i}.png"))
for i,r in enumerate(mismatch):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_mismatch_{i}.png"))
for i,r in enumerate(size_results):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_size_{i}.png"))
