#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from funs import funs

script_name="2a_combinedfigures"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
alpha=0.1
beta=0.02
gamma=0.02
settling_steps=300

configs_size=[
    {"name":"8x8\np_line=1/8, p=1/8", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"16x16\np_line=1/16, p=1/16", "size":(16,16), "p_line":1/16, "p":1/16, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":2500, "settling_steps":settling_steps},
    {"name":"16x16\np_line=1/8, p=1/8", "size":(16,16), "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":2500, "settling_steps":settling_steps},
]
configs_matched=[
    {"name":"matched\n1/8", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"matched\n1/4", "size":(8,8), "p_line":1/4, "p":1/4, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"matched\n1/2", "size":(8,8), "p_line":1/2, "p":1/2, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":1500, "settling_steps":settling_steps},
]
configs_mismatch=[
    {"name":"fixed p\np_line=1/8", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"fixed p\np_line=1/4", "size":(8,8), "p_line":1/4, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"fixed p\np_line=1/2", "size":(8,8), "p_line":1/2, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":1500, "settling_steps":settling_steps},
]
configs_lr=[
    {"name":"x0.25", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":0.025, "beta":0.005, "gamma":0.005, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"x0.5", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":0.05, "beta":0.01, "gamma":0.01, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"x1", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":0.1, "beta":0.02, "gamma":0.02, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"x2", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":0.2, "beta":0.04, "gamma":0.04, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"x4", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":0.4, "beta":0.08, "gamma":0.08, "num_batches":1500, "settling_steps":settling_steps},
    {"name":"x8", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":0.8, "beta":0.16, "gamma":0.16, "num_batches":1500, "settling_steps":settling_steps},
]

#%%
size_results=funs.run_configs(configs_size)
matched_results=funs.run_configs(configs_matched)
mismatch_results=funs.run_configs(configs_mismatch)
lr_results=funs.run_configs(configs_lr)

#%%
def plot_combined_bars(groups, save_path):
    colors={"score":"#4f6fd5", "coverage":"0.58"}
    fig,axes=plt.subplots(1,len(groups),figsize=(14.5,3.6),dpi=240,facecolor="white",gridspec_kw={"width_ratios":[3,3,3,6]})
    for ax,(title,results) in zip(axes,groups):
        x=np.arange(len(results)); scores=[r["score"] for r in results]; coverages=[r["coverage"] for r in results]
        ax.bar(x-0.18,scores,width=0.36,color=colors["score"],edgecolor="white",lw=.8,alpha=.9,label="line score")
        ax.bar(x+0.18,coverages,width=0.36,color=colors["coverage"],edgecolor="white",lw=.8,alpha=.82,label="coverage")
        ax.axhline(0.8,color="#d62728",lw=1,ls="--",alpha=.7)
        ax.set_title(title,fontsize=11,pad=8)
        ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in results],rotation=25,ha="right",fontsize=7.5)
        ax.set_ylim(0,1.05); ax.grid(axis="y",color="0.9",lw=.8); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("score")
    axes[-1].legend(frameon=False,loc="upper right",fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.show()

plot_combined_bars([("image size",size_results),("matched p",matched_results),("p fixed at 1/8",mismatch_results),("learning-rate scale",lr_results)], os.path.join(save_dir,f"combined_bars.png"))

#%%
def plot_activity_panels(groups, save_path):
    colors=["#1b1b1b","#4f6fd5","#d62728","#6b6b6b","#f58518","#54a24b","#b279a2","#72b7b2"]
    fig,axes=plt.subplots(1,len(groups),figsize=(13.5,3.6),dpi=240,facecolor="white",sharey=True)
    for ax,(title,results) in zip(axes,groups):
        for i,r in enumerate(results):
            ax.plot(np.convolve(r["activity"],np.ones(80)/80,mode="valid"),color=colors[i%len(colors)],lw=1.55,label=r["name"].replace("\n"," "))
        ax.axhline(results[0]["p"],color="0.45",lw=.9,ls="--",alpha=.85)
        ax.set_title(title,fontsize=11,pad=8); ax.set_xlabel("training step")
        ax.grid(color="0.9",lw=.8); ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=6.4,loc="upper right")
    axes[0].set_ylabel("mean activity")
    plt.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.show()

plot_activity_panels([("image size",size_results),("matched p",matched_results),("p fixed at 1/8",mismatch_results),("learning-rate scale",lr_results)], os.path.join(save_dir,f"activity_panels.png"))

#%%
def plot_weight_montage(results, title, save_path):
    ncols=8
    nrows=sum(int(np.ceil(r["qij"].shape[0]/ncols)) for r in results)
    fig,axes=plt.subplots(nrows,ncols,figsize=(10.8,1.28*nrows),dpi=240,facecolor="white")
    axes=np.array(axes).reshape(nrows,ncols)
    row0=0
    for r in results:
        qij=r["qij"]; size=r["size"]; rows=int(np.ceil(qij.shape[0]/ncols)); vmin=np.percentile(qij,2); vmax=np.percentile(qij,98)
        for k in range(rows*ncols):
            ax=axes[row0+k//ncols,k%ncols]; ax.axis("off")
            if k<qij.shape[0]:
                ax.imshow(qij[k].reshape(size),cmap="gray",interpolation="nearest",vmin=vmin,vmax=vmax)
                ax.add_patch(Rectangle((-0.5,-0.5),size[1],size[0],fill=False,edgecolor="#d62728",linewidth=.75))
                ax.set_title(f"{k}: {r['best'][k]:.2f}",fontsize=5.8,pad=1.5)
        axes[row0,0].text(-0.35,0.5,r["name"].replace("\n"," "),transform=axes[row0,0].transAxes,rotation=90,ha="right",va="center",fontsize=8)
        row0+=rows
    fig.suptitle(title,fontsize=13,y=0.998)
    plt.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.show()

plot_weight_montage(size_results, "learned feedforward weights: image size conditions", os.path.join(save_dir,f"weights_size.png"))
plot_weight_montage([matched_results[0],matched_results[2],mismatch_results[2]], "learned feedforward weights: sparse baseline vs dense inputs", os.path.join(save_dir,f"weights_dense_examples.png"))
