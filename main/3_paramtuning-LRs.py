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
batch_size=1
base_alpha=0.1
base_beta=0.02
base_gamma=0.02
scales=[0.001,0.01,0.1,0.25,0.5,1,2,4,8,10,100]

def make_configs(mode):
    configs=[]
    for s in scales:
        alpha=base_alpha*s if mode in ["all","alpha"] else base_alpha
        beta=base_beta*s if mode in ["all","beta"] else base_beta
        gamma=base_gamma*s if mode in ["all","gamma"] else base_gamma
        configs.append({"name":f"x{s:g}", "size":size, "p_line":p_line, "p":p, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size, "gammatuned":mode=="gamma"})
    return configs

#%%
groups={
    "all rates scaled":funs.run_configs(make_configs("all"), show_note=True),
    "alpha only":funs.run_configs(make_configs("alpha"), show_note=True),
    "beta only":funs.run_configs(make_configs("beta"), show_note=True),
    "gamma only":funs.run_configs(make_configs("gamma"), show_note=True),
}

#%%
fig,axes=plt.subplots(2,2,figsize=(13,6.8),dpi=220,facecolor="white",sharex=True,sharey=True)
colors={"line score":"#4c78a8", "coverage":"#54a24b", "mean activity":"#f58518"}
markers={"line score":"o", "coverage":"s", "mean activity":"^"}
major_ticks=[0.001,0.01,0.1,1,10,100]
for ax,(title,results) in zip(axes.reshape(-1),groups.items()):
    scores=[r["score"] for r in results]
    coverages=[r["coverage"] for r in results]
    activities=[np.mean(r["activity"][-200:]) for r in results]
    errors={"line score":[funs.score_sem(r) for r in results], "coverage":None, "mean activity":[funs.activity_sem(r) for r in results]}
    for label,values in {"line score":scores, "coverage":coverages, "mean activity":activities}.items():
        ax.errorbar(scales, values, yerr=errors[label], marker=markers[label], lw=1.8, ms=4.5, capsize=2.2, color=colors[label], label=label)
    ax.set_xscale("log", base=10)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"x{s:g}" for s in major_ticks])
    ax.set_ylim(0,1.08)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="0.9", lw=.8)
    ax.spines[["top","right"]].set_visible(False)
for ax in axes[:,0]:
    ax.set_ylabel("value")
for ax in axes[-1,:]:
    ax.set_xlabel("learning rate scale")
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,frameon=False,ncols=3,loc="upper center",bbox_to_anchor=(0.5,0.96))
fig.suptitle("learning rate tests", y=0.995, fontsize=13)
plt.tight_layout(rect=[0,0,1,0.92])
fig.savefig(os.path.join(save_dir, f"{script_name}_summary_scale.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

#%%
for i,r in enumerate(groups["all rates scaled"]):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_scale_{i}.png"), show=False)
