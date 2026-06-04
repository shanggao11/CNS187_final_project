#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
batch_size=1

configs_matched=[
    {"name":"p_line=1/8, p=1/8", "size":size, "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size},
    {"name":"p_line=1/4, p=1/4", "size":size, "p_line":1/4, "p":1/4, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size},
    {"name":"p_line=1/2, p=1/2", "size":size, "p_line":1/2, "p":1/2, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size},
]

configs_mismatch=[
    {"name":"p_line=1/8, p=1/8", "size":size, "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size},
    {"name":"p_line=1/4, p=1/8", "size":size, "p_line":1/4, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size},
    {"name":"p_line=1/2, p=1/8", "size":size, "p_line":1/2, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size},
]


configs_mismatch2=[
    {"name":"p_line=1/8, p=1/8", "size":size, "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size},
    {"name":"p_line=1/8, p=1/4", "size":size, "p_line":1/8, "p":1/4, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size},
    {"name":"p_line=1/8, p=1/2", "size":size, "p_line":1/8, "p":1/2, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":num_batches, "settling_steps":settling_steps, "batch_size":batch_size},
]


configs_size=[
    {"name":"8x8, p_line=1/8, p=1/8", "size":(8,8), "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":1500, "settling_steps":settling_steps, "batch_size":batch_size},
    {"name":"16x16, p_line=1/16, p=1/16", "size":(16,16), "p_line":1/16, "p":1/16, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":2500, "settling_steps":settling_steps, "batch_size":batch_size},
    {"name":"16x16, p_line=1/16, p=1/8", "size":(16,16), "p_line":1/16, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":2500, "settling_steps":settling_steps, "batch_size":batch_size},
    {"name":"16x16, p_line=1/8, p=1/8", "size":(16,16), "p_line":1/8, "p":1/8, "alpha":alpha, "beta":beta, "gamma":gamma, "num_batches":2500, "settling_steps":settling_steps, "batch_size":batch_size},
]

#%%
matched=funs.run_configs(configs_matched)
mismatch=funs.run_configs(configs_mismatch)
mismatch2=funs.run_configs(configs_mismatch2)
size_results=funs.run_configs(configs_size)

#%%
funs.plot_summary(matched, "p_line and p both changed", os.path.join(save_dir, f"{script_name}_summary_matched.png"))
funs.plot_summary(mismatch, "p_line changed, p fixed at 1/8", os.path.join(save_dir, f"{script_name}_summary_mismatch.png"))
funs.plot_summary(mismatch2, "p_line fixed at 1/8, p changed", os.path.join(save_dir, f"{script_name}_summary_mismatch2.png"))
funs.plot_summary(size_results, "image size experiments", os.path.join(save_dir, f"{script_name}_summary_size.png"))

#%%
funs.plot_activity(matched, "activity: matched p", os.path.join(save_dir, f"{script_name}_activity_matched.png"))
funs.plot_activity(mismatch, "activity: mismatched p", os.path.join(save_dir, f"{script_name}_activity_mismatch.png"))
funs.plot_activity(mismatch2, "activity: mismatch2 p", os.path.join(save_dir, f"{script_name}_activity_mismatch2.png"))
funs.plot_activity(size_results, "activity: image size", os.path.join(save_dir, f"{script_name}_activity_size.png"))

#%%
for i,r in enumerate(matched):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_matched_{i}.png"))
for i,r in enumerate(mismatch):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_mismatch_{i}.png"))
for i,r in enumerate(mismatch2):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_mismatch2_{i}.png"))
for i,r in enumerate(size_results):
    funs.plot_qij(r, save_path=os.path.join(save_dir, f"{script_name}_qij_size_{i}.png"))

#%%
def plot_combined_lines(groups, save_path):
    fig,axes=plt.subplots(2,2,figsize=(9.5,6),dpi=220,facecolor="white",sharey=True)
    colors={"line score":"#4c78a8","coverage":"#54a24b"}
    for ax,(title,results) in zip(axes.reshape(-1),groups):
        x=np.arange(len(results))
        ax.errorbar(x,[r["score"] for r in results],yerr=[funs.score_sem(r) for r in results],marker="o",lw=1.8,capsize=2.4,color=colors["line score"],label="line score")
        ax.plot(x,[r["coverage"] for r in results],marker="s",lw=1.8,color=colors["coverage"],label="coverage")
        ax.axhline(.8,color="#d62728",lw=.9,ls="--",alpha=.7)
        ax.set_title(title,fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([r["name"].replace(", ","\n") for r in results],fontsize=7)
        ax.set_ylim(0,1.05)
        ax.grid(axis="y",color="0.9",lw=.8)
        ax.spines[["top","right"]].set_visible(False)
    for ax in axes[:,0]:
        ax.set_ylabel("score")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncols=2,loc="upper center",bbox_to_anchor=(0.5,1.02))
    plt.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    

def plot_activity_panels(groups, save_path):
    colors=["#1b1b1b","#4c78a8","#d62728","#54a24b"]
    fig,axes=plt.subplots(2,2,figsize=(9.5,6),dpi=220,facecolor="white",sharey=True)
    for ax,(title,results) in zip(axes.reshape(-1),groups):
        for i,r in enumerate(results):
            ax.plot(np.convolve(r["activity"],np.ones(80)/80,mode="valid"),color=colors[i%len(colors)],lw=1.45,label=r["name"])
        ax.axhline(results[0]["p"],color="0.45",lw=.9,ls="--",alpha=.85)
        ax.set_title(title,fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(color="0.9",lw=.8)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=6.2,loc="upper right")
    for ax in axes[:,0]:
        ax.set_ylabel("mean activity")
    plt.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    

def plot_weight_montage(results, title, save_path):
    ncols=8
    nrows=sum(int(np.ceil(r["qij"].shape[0]/ncols)) for r in results)
    fig,axes=plt.subplots(nrows,ncols,figsize=(10.8,1.28*nrows),dpi=220,facecolor="white")
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
        axes[row0,0].text(-0.35,0.5,r["name"],transform=axes[row0,0].transAxes,rotation=90,ha="right",va="center",fontsize=8)
        row0+=rows
    fig.suptitle(title,fontsize=13,y=.998)
    plt.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    

groups=[("image size",size_results),("matched p",matched),("p fixed at 1/8",mismatch),("p_line fixed at 1/8",mismatch2)]
plot_combined_lines(groups, os.path.join(save_dir,f"{script_name}_combined_lines.png"))
plot_activity_panels(groups, os.path.join(save_dir,f"{script_name}_activity_panels.png"))
plot_weight_montage(size_results, "learned feedforward weights: image size conditions", os.path.join(save_dir,f"{script_name}_weights_size.png"))
plot_weight_montage([matched[0],matched[2],mismatch[2]], "learned feedforward weights: sparse baseline vs dense inputs", os.path.join(save_dir,f"{script_name}_weights_dense_examples.png"))
