#%%
import numpy as np
import matplotlib.pyplot as plt
from funs import funs
import os

script_name="3_paramtuning-LRs"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
size=(8,8)
p_line=1/8
p=1/8
num_batches=1500
settling_steps=300
configs_scale=[
    {"name":"x0.25", "size":size, "p_line":p_line, "p":p, "alpha":0.025, "beta":0.005, "gamma":0.005, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"x0.5", "size":size, "p_line":p_line, "p":p, "alpha":0.05, "beta":0.01, "gamma":0.01, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"x1", "size":size, "p_line":p_line, "p":p, "alpha":0.1, "beta":0.02, "gamma":0.02, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"x2", "size":size, "p_line":p_line, "p":p, "alpha":0.2, "beta":0.04, "gamma":0.04, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"x4", "size":size, "p_line":p_line, "p":p, "alpha":0.4, "beta":0.08, "gamma":0.08, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"x8", "size":size, "p_line":p_line, "p":p, "alpha":0.8, "beta":0.16, "gamma":0.16, "num_batches":num_batches, "settling_steps":settling_steps},
]

configs_single=[
    {"name":"higher alpha only", "size":size, "p_line":p_line, "p":p, "alpha":0.4, "beta":0.02, "gamma":0.02, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"higher beta only", "size":size, "p_line":p_line, "p":p, "alpha":0.1, "beta":0.08, "gamma":0.02, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"higher gamma only", "size":size, "p_line":p_line, "p":p, "alpha":0.1, "beta":0.02, "gamma":0.08, "num_batches":num_batches, "settling_steps":settling_steps},
    {"name":"lower beta only", "size":size, "p_line":p_line, "p":p, "alpha":0.1, "beta":0.005, "gamma":0.02, "num_batches":num_batches, "settling_steps":settling_steps},
]

#%%
results_scale=funs.run_configs(configs_scale, show_note=True)
results_single=funs.run_configs(configs_single, show_note=True)

#%%
funs.plot_summary(results_scale, "learning rates scaled together", os.path.join(save_dir, f"{script_name}_summary_scale.png"))
funs.plot_summary(results_single, "one learning rate changed", os.path.join(save_dir, f"{script_name}_summary_single.png"))
funs.plot_activity(results_scale, "activity: scaled learning rates", os.path.join(save_dir, f"{script_name}_activity_scale.png"))
funs.plot_activity(results_single, "activity: single learning rate changes", os.path.join(save_dir, f"{script_name}_activity_single.png"))

#%%
scales=[]
scores=[]
activities=[]
for r in results_scale:
    scales.append(r["alpha"]/0.1)
    scores.append(r["score"])
    activities.append(np.mean(r["activity"][-200:]))

fig,ax1=plt.subplots(figsize=(7,3.2), facecolor="white")
ax1.plot(scales, scores, marker="o", linewidth=2, color="#4c78a8", label="line score")
ax1.set_xscale("log", base=2)
ax1.set_xticks(scales)
ax1.set_xticklabels([f"x{s:g}" for s in scales])
ax1.set_ylim(0,1.05)
ax1.set_xlabel("learning rate scale")
ax1.set_ylabel("line score")
ax2=ax1.twinx()
ax2.plot(scales, activities, marker="s", linewidth=2, color="#f58518", label="mean activity")
ax2.set_ylabel("mean activity")
ax1.legend(frameon=False, loc="upper left")
ax2.legend(frameon=False, loc="upper right")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_score_activity_curve.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
for i,r in enumerate(results_scale):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_scale_{i}.png"))
for i,r in enumerate(results_single):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_single_{i}.png"))
