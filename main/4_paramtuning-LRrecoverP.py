#%%
import numpy as np
from funs import funs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
script_name="4_paramtuning-LRrecoverP"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
size=(8,8)
num_batches=1500
settling_steps=300
batch_size=1
base_alpha=0.1
base_beta=0.02
base_gamma=0.02
scales=[0.001,0.01,0.1,0.25,0.5,1,2,4,8,10,100]
modes=["all","alpha","beta","gamma"]

baseline_name="baseline p_line=1/8, p=1/8"
p_names=[baseline_name, "matched p_line=1/4", "matched p_line=1/2", "mismatch p_line=1/4, p=1/8", "mismatch p_line=1/2, p=1/8"]
p_lines=[1/8, 1/4, 1/2, 1/4, 1/2]
ps=[1/8, 1/4, 1/2, 1/8, 1/8]
lr_names=[f"{mode} x{s:g}" for mode in modes for s in scales]

configs=[]
for i in range(len(p_names)):
    for mode in modes:
        for s in scales:
            alpha=base_alpha*s if mode in ["all","alpha"] else base_alpha
            beta=base_beta*s if mode in ["all","beta"] else base_beta
            gamma=base_gamma*s if mode in ["all","gamma"] else base_gamma
            name=p_names[i]+f" | {mode} x{s:g}"
            configs.append({"name":name, "size":size, "p_line":p_lines[i], "p":ps[i], "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size, "gammatuned":mode=="gamma"})

#%%
results=funs.run_configs(configs, show_note=True)

#%%
baseline_result=next(r for r in results if r["name"]==baseline_name+" | all x1")
baseline_score=baseline_result["score"]
baseline_coverage=baseline_result["coverage"]
baseline_activity=np.mean(baseline_result["activity"][-200:])
print(f"baseline reference: score={baseline_score:.3f}, coverage={baseline_coverage:.3f}, activity={baseline_activity:.3f}")
recovery_names=[name for name in p_names if name!=baseline_name]
best_results=[]
for i in range(len(recovery_names)):
    best=None
    for r in results:
        if r["name"].startswith(recovery_names[i]):
            if best is None or r["score"]>best["score"]:
                best=r
    best_results.append(best)
    lr_name=best["name"].split("|")[-1].strip()
    print(f"best for {recovery_names[i]}: {lr_name}, alpha={best['alpha']}, beta={best['beta']}, gamma={best['gamma']}, score={best['score']:.3f}, recovery={best['score']/baseline_score:.2f}, coverage={best['coverage']:.3f}, note={funs.learning_rate_note(best)}")

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

fig,ax=plt.subplots(figsize=(max(14,.34*len(lr_names)),4.2), facecolor="white")
im=ax.imshow(score_mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(np.arange(len(lr_names)))
ax.set_xticklabels(lr_names, rotation=90, ha="center", fontsize=6)
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
# if not os.path.exists(save_path):
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

#%%
fig,ax=plt.subplots(figsize=(max(14,.34*len(lr_names)),4.2), facecolor="white")
im=ax.imshow(activity_mat, cmap="magma", aspect="auto")
ax.set_xticks(np.arange(len(lr_names)))
ax.set_xticklabels(lr_names, rotation=90, ha="center", fontsize=6)
ax.set_yticks(np.arange(len(p_names)))
ax.set_yticklabels(p_names)
for i in range(activity_mat.shape[0]):
    for j in range(activity_mat.shape[1]):
        ax.text(j,i,f"{activity_mat[i,j]:.2f}",ha="center",va="center",color="white",fontsize=8)
ax.set_title("Mean activity for each recovery condition")
fig.colorbar(im, ax=ax, label="mean activity")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_activity_heatmap.png")
# if not os.path.exists(save_path):
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

#%%
colors={"all":"#4c78a8","alpha":"#54a24b","beta":"#f58518","gamma":"#b279a2"}
fig,axes=plt.subplots(2,2,figsize=(11,6.8),dpi=220,facecolor="white",sharex=True,sharey=True)
for ax,condition in zip(axes.reshape(-1),recovery_names):
    for mode in modes:
        score_vals=[]
        score_err=[]
        coverage_vals=[]
        for s in scales:
            name=condition+f" | {mode} x{s:g}"
            result=next(r for r in results if r["name"]==name)
            score_vals.append(result["score"]/baseline_score)
            score_err.append(funs.score_sem(result)/baseline_score)
            coverage_vals.append(result["coverage"]/baseline_coverage)
        ax.errorbar(scales,score_vals,yerr=score_err,marker="o",ms=3.5,lw=1.7,capsize=2.1,color=colors[mode],label=mode)
        ax.plot(scales,coverage_vals,marker="s",ms=3,lw=1.4,ls="--",color=colors[mode],alpha=.75)
    ax.axhline(1,color="0.25",lw=1,ls="--",alpha=.75)
    ax.set_xscale("log",base=10)
    ax.set_xticks([0.001,0.01,0.1,1,10,100])
    ax.set_xticklabels(["x0.001","x0.01","x0.1","x1","x10","x100"])
    ax.set_ylim(0,1.05)
    ax.set_title(condition,fontsize=9)
    ax.grid(axis="y",color="0.9",lw=.8)
    ax.spines[["top","right"]].set_visible(False)
for ax in axes[:,0]:
    ax.set_ylabel("recovery ratio")
for ax in axes[-1,:]:
    ax.set_xlabel("learning rate scale")
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,frameon=False,ncols=4,loc="upper center",bbox_to_anchor=(0.5,0.97))
style_handles=[Line2D([0],[0],color="0.25",lw=1.7,marker="o",label="line score"), Line2D([0],[0],color="0.25",lw=1.4,ls="--",marker="s",label="coverage")]
fig.legend(handles=style_handles,frameon=False,ncols=2,loc="upper center",bbox_to_anchor=(0.5,0.925))
fig.suptitle("Recovery relative to baseline p_line=p=1/8",y=0.995,fontsize=13)
plt.tight_layout(rect=[0,0,1,0.88])
fig.savefig(os.path.join(save_dir, f"{script_name}_recovery_ratio_lines.png"),dpi=300,bbox_inches="tight")
plt.close(fig)

#%%
for i,r in enumerate(best_results):
    funs.plot_qij(r, f"best recovery: {r['name']} | score={r['score']:.3f}, coverage={r['coverage']:.3f}", os.path.join(save_dir, f"{script_name}_best_qij_{i}.png"), show=False)
