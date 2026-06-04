#%%
import numpy as np
import matplotlib.pyplot as plt
from funs import funs
import os
import csv

script_name="10_falconbridge_feature_regime"
source_dir=os.path.join("Results","9_falconbridge_first_foldiak")
bruno_paths=[os.path.join("Results","7_bruno_sparsecoding",f"Bruno_BASIS1_NUM_{n}_size16.npy") for n in [256,2048]]
save_dir=os.path.join("Results",script_name)
os.makedirs(save_dir,exist_ok=True)
xdim=16
ydim=16
patch_dim=xdim*ydim
plt.rcParams.update({"figure.dpi":140,"axes.spines.top":False,"axes.spines.right":False,"font.size":9})

runs=funs.load_falconbridge_feature_runs(source_dir, bruno_paths, patch_dim)
for r in runs:
    rf=r["rf"]
    q=rf-np.mean(rf,axis=1,keepdims=True); q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12); sim=np.abs(q@q.T); mask=~np.eye(sim.shape[0],dtype=bool)
    r["rf_concentration_sem"]=np.std(funs.top_energy_fraction(rf,average=False),ddof=1)/np.sqrt(rf.shape[0])
    r["mean_cosine_sem"]=np.std(sim[mask].reshape(rf.shape[0],-1).mean(axis=1),ddof=1)/np.sqrt(rf.shape[0])
    r["nn90_cosine_sem"]=np.std(np.max(sim-np.eye(sim.shape[0]),axis=1),ddof=1)/np.sqrt(rf.shape[0])
metrics_path=os.path.join(save_dir,f"{script_name}_metrics.csv")
cols=["name","sparsity","rf_concentration","rf_concentration_sem","mean_cosine","mean_cosine_sem","nn90_cosine","nn90_cosine_sem","output_corr","dead_units"]
with open(metrics_path,"w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=cols); writer.writeheader(); writer.writerows([{k:r[k] for k in cols} for r in runs])

#%%
names=[r["name"].replace("N","N=") for r in runs]
x=np.arange(len(runs))
fig,axes=plt.subplots(2,1,figsize=(3.7,4.4),dpi=220,facecolor="white")
ax=axes[0]
ax.bar(x,[r["rf_concentration"] for r in runs],yerr=[r["rf_concentration_sem"] for r in runs],capsize=2.4,color="#2ca02c",edgecolor="white",linewidth=.8)
ax.set_title("RF localization",fontsize=10); ax.set_ylabel("top 10% energy")
ax=axes[1]
w=.34
ax.bar(x-w/2,[r["mean_cosine"] for r in runs],width=w,yerr=[r["mean_cosine_sem"] for r in runs],capsize=2.4,color="0.45",edgecolor="white",linewidth=.8,label="mean")
ax.bar(x+w/2,[r["nn90_cosine"] for r in runs],width=w,yerr=[r["nn90_cosine_sem"] for r in runs],capsize=2.4,color="#9467bd",edgecolor="white",linewidth=.8,label="near-copy")
ax.set_title("redundancy",fontsize=10); ax.set_ylabel("|cosine|"); ax.legend(frameon=False,fontsize=7)
for ax in axes.reshape(-1):
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=15,ha="right",fontsize=8)
plt.tight_layout(h_pad=.9)
fig.savefig(os.path.join(save_dir,f"{script_name}_summary.png"),dpi=300,bbox_inches="tight")
plt.show()
