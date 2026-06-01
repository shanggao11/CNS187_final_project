#%%
import numpy as np
from funs import funs
import matplotlib.pyplot as plt
import os

script_name="4_paramtuning-LRrecoverP"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
size=(8,8)
num_batches=1500
settling_steps=300

p_names=["matched p_line=1/4", "matched p_line=1/2", "mismatch p_line=1/4, p=1/8", "mismatch p_line=1/2, p=1/8"]
p_lines=[1/4, 1/2, 1/4, 1/2]
ps=[1/4, 1/2, 1/8, 1/8]

lr_names=["slow", "baseline", "fast", "faster", "high alpha", "high beta", "high gamma", "low beta"]
alphas=[0.05, 0.1, 0.2, 0.4, 0.3, 0.1, 0.1, 0.1]
betas=[0.01, 0.02, 0.04, 0.08, 0.02, 0.08, 0.02, 0.005]
gammas=[0.01, 0.02, 0.04, 0.08, 0.02, 0.02, 0.08, 0.02]

configs=[]
for i in range(len(p_names)):
    for j in range(len(lr_names)):
        name=p_names[i]+" | "+lr_names[j]
        configs.append({"name":name, "size":size, "p_line":p_lines[i], "p":ps[i], "alpha":alphas[j], "beta":betas[j], "gamma":gammas[j], "num_batches":num_batches, "settling_steps":settling_steps})

#%%
results=funs.run_configs(configs, show_note=True)

#%%
best_results=[]
for i in range(len(p_names)):
    best=None
    for r in results:
        if r["name"].startswith(p_names[i]):
            if best is None or r["score"]>best["score"]:
                best=r
    best_results.append(best)
    lr_name=best["name"].split("|")[-1].strip()
    print(f"best for {p_names[i]}: {lr_name}, alpha={best['alpha']}, beta={best['beta']}, gamma={best['gamma']}, score={best['score']:.3f}, coverage={best['coverage']:.3f}, note={funs.learning_rate_note(best)}")

#%%
score_mat=np.zeros((len(p_names),len(lr_names)))
activity_mat=np.zeros((len(p_names),len(lr_names)))
for i in range(len(p_names)):
    for j in range(len(lr_names)):
        name=p_names[i]+" | "+lr_names[j]
        for r in results:
            if r["name"]==name:
                score_mat[i,j]=r["score"]
                activity_mat[i,j]=np.mean(r["activity"][-200:])

fig,ax=plt.subplots(figsize=(10,3.8), facecolor="white")
im=ax.imshow(score_mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(np.arange(len(lr_names)))
ax.set_xticklabels(lr_names, rotation=25, ha="right")
ax.set_yticks(np.arange(len(p_names)))
ax.set_yticklabels(p_names)
for i in range(score_mat.shape[0]):
    for j in range(score_mat.shape[1]):
        color="white" if score_mat[i,j]<0.65 else "black"
        ax.text(j,i,f"{score_mat[i,j]:.2f}",ha="center",va="center",color=color,fontsize=8)
ax.set_title("Can learning rates recover line detectors after changing p?")
fig.colorbar(im, ax=ax, label="line score")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_score_heatmap.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
fig,ax=plt.subplots(figsize=(10,3.8), facecolor="white")
im=ax.imshow(activity_mat, cmap="magma", aspect="auto")
ax.set_xticks(np.arange(len(lr_names)))
ax.set_xticklabels(lr_names, rotation=25, ha="right")
ax.set_yticks(np.arange(len(p_names)))
ax.set_yticklabels(p_names)
for i in range(activity_mat.shape[0]):
    for j in range(activity_mat.shape[1]):
        ax.text(j,i,f"{activity_mat[i,j]:.2f}",ha="center",va="center",color="white",fontsize=8)
ax.set_title("Mean activity for each recovery condition")
fig.colorbar(im, ax=ax, label="mean activity")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_activity_heatmap.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
for i,r in enumerate(best_results):
    funs.plot_qij(r, f"best recovery: {r['name']} | score={r['score']:.3f}, coverage={r['coverage']:.3f}", os.path.join(save_dir, f"{script_name}_best_qij_{i}.png"))
