#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from funs import funs
import os
import csv

script_name="6_mnist_foldiak"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
np.random.seed(0)
torch.manual_seed(0)
resolutions=[14,10]
output_sizes=[64,128]
train_n=3000
test_n=1000
num_batches=train_n
settling_steps=120
alpha=0.08
beta=0.02
gamma=0.02
p=0.1
p2=0.14
dt=0.01
lambda_=10
pretrain_patterns=100
batch_size=1
results_path=os.path.join(save_dir, f"{script_name}_results.npy")
#%%
train_loader,test_loader=funs.prepare_mnist_data(batch_size=256)
train_x,train_y=[],[]
test_x,test_y=[],[]
for x,y in train_loader:
    train_x.append(x); train_y.append(y)
    if len(torch.cat(train_y))>=train_n:
        break
for x,y in test_loader:
    test_x.append(x); test_y.append(y)
    if len(torch.cat(test_y))>=test_n:
        break

train_x=torch.cat(train_x)[:train_n] # (3000, 1, 28, 28)
train_y=torch.cat(train_y)[:train_n]# (3000,)
test_x=torch.cat(test_x)[:test_n] # (1000, 1, 28, 28)
test_y=torch.cat(test_y)[:test_n] # (1000,)

if os.path.exists(results_path):
    results=np.load(results_path, allow_pickle=True).tolist()
    if not all("digit_selectivity_units" in r for r in results):
        results=[]
else:
    results=[]
if len(results)==0:
    for resolution in resolutions:
        x_train=funs.resize_flat(train_x, resolution)
        x_test=funs.resize_flat(test_x, resolution)
        raw_train_acc,raw_test_acc=funs.perceptron_accuracy(x_train, train_y.numpy(), x_test, test_y.numpy())
        for number_of_outputs in output_sizes:
            qij,wij,ti,activity=funs.train_foldiak(x_train[:num_batches], number_of_outputs, p, alpha, beta, gamma, lambda_, dt, settling_steps, num_batches, pretrain_patterns,batch_size=batch_size)
            z_train=funs.encode_foldiak(x_train, qij, wij, ti, lambda_, dt, settling_steps)
            z_test=funs.encode_foldiak(x_test, qij, wij, ti, lambda_, dt, settling_steps)
            sparse_train_acc,sparse_test_acc=funs.perceptron_accuracy(z_train, train_y.numpy(), z_test, test_y.numpy())
            layer2_output_size=number_of_outputs//2
            q2,w2,t2,activity2=funs.train_foldiak(z_train[:num_batches], layer2_output_size, p2, alpha, beta, gamma, lambda_, dt, settling_steps, num_batches, pretrain_patterns,batch_size=batch_size)
            z2_train=funs.encode_foldiak(z_train, q2, w2, t2, lambda_, dt, settling_steps)
            z2_test=funs.encode_foldiak(z_test, q2, w2, t2, lambda_, dt, settling_steps)
            layer2_train_acc,layer2_test_acc=funs.perceptron_accuracy(z2_train, train_y.numpy(), z2_test, test_y.numpy())
            both_train_acc,both_test_acc=funs.perceptron_accuracy(np.c_[z_train,z2_train], train_y.numpy(), np.c_[z_test,z2_test], test_y.numpy())
            rf1_sta=funs.activity_rf(x_train, z_train)
            rf2_sta=funs.activity_rf(x_train, z2_train)
            qn=qij/(np.linalg.norm(qij,axis=1,keepdims=True)+1e-12); rn=rf1_sta/(np.linalg.norm(rf1_sta,axis=1,keepdims=True)+1e-12); mask=~np.eye(qij.shape[0],dtype=bool)
            result={"resolution":resolution, "number_of_outputs":number_of_outputs, "layer2_output_size":layer2_output_size, "raw_test_acc":raw_test_acc, "sparse_test_acc":sparse_test_acc, "layer2_test_acc":layer2_test_acc, "both_test_acc":both_test_acc, "sparsity":np.mean(z_train), "layer2_sparsity":np.mean(z2_train), "sparsity_units":np.mean(z_train,axis=0), "layer2_sparsity_units":np.mean(z2_train,axis=0), "filter_similarity":funs.mean_abs_offdiag_cosine(qij), "sta_similarity":funs.mean_abs_offdiag_cosine(rf1_sta), "filter_similarity_units":np.mean(np.abs(qn@qn.T)[mask].reshape(qij.shape[0],-1),axis=1), "sta_similarity_units":np.mean(np.abs(rn@rn.T)[mask].reshape(rf1_sta.shape[0],-1),axis=1), "digit_selectivity":funs.digit_selectivity(z_train,train_y.numpy()), "layer2_digit_selectivity":funs.digit_selectivity(z2_train,train_y.numpy()), "rf_top10_energy":funs.top_energy_fraction(rf1_sta), "layer2_rf_top10_energy":funs.top_energy_fraction(rf2_sta), "rf_top10_energy_units":funs.top_energy_fraction(rf1_sta,average=False), "layer2_rf_top10_energy_units":funs.top_energy_fraction(rf2_sta,average=False), "activity":activity, "activity2":activity2, "qij":qij, "q2":q2, "rf1_sta":rf1_sta, "rf2_sta":rf2_sta}
            responses1=np.vstack([np.mean(z_train[train_y.numpy()==d],axis=0) for d in range(10)])
            responses2=np.vstack([np.mean(z2_train[train_y.numpy()==d],axis=0) for d in range(10)])
            result["digit_selectivity_units"]=np.max(responses1,axis=0)/(np.mean(responses1,axis=0)+1e-12)
            result["layer2_digit_selectivity_units"]=np.max(responses2,axis=0)/(np.mean(responses2,axis=0)+1e-12)
            results.append(result)
            print(f"res={resolution}, outputs={number_of_outputs}, raw_acc={raw_test_acc:.3f}, layer1_acc={sparse_test_acc:.3f}, layer2_acc={layer2_test_acc:.3f}, both_acc={both_test_acc:.3f}, layer1_activity={np.mean(z_train):.3f}, layer2_activity={np.mean(z2_train):.3f}")
    np.save(results_path, np.array(results, dtype=object), allow_pickle=True)
summary_cols=["resolution","number_of_outputs","layer2_output_size","sparsity","layer2_sparsity","raw_test_acc","sparse_test_acc","layer2_test_acc","both_test_acc","filter_similarity","sta_similarity","digit_selectivity","layer2_digit_selectivity","rf_top10_energy","layer2_rf_top10_energy"]
with open(os.path.join(save_dir, f"{script_name}_summary_metrics.csv"), "w", newline="") as f:
    writer=csv.DictWriter(f, fieldnames=summary_cols); writer.writeheader(); writer.writerows([{k:r.get(k,np.nan) for k in summary_cols} for r in results])

#%%
names=[f"{r['resolution']}px/{r['number_of_outputs']}" for r in results]
x=np.arange(len(results))
raw_acc=[r["raw_test_acc"] for r in results]
sparse_acc=[r["sparse_test_acc"] for r in results]
layer2_acc=[r["layer2_test_acc"] for r in results]
both_acc=[r["both_test_acc"] for r in results]
sparsity=[r["sparsity"] for r in results]
layer2_sparsity=[r["layer2_sparsity"] for r in results]
filter_similarity=[r["filter_similarity"] for r in results]
sta_similarity=[r["sta_similarity"] for r in results]
digit_selectivity=[r["digit_selectivity"] for r in results]
layer2_digit_selectivity=[r["layer2_digit_selectivity"] for r in results]
rf_top10_energy=[r["rf_top10_energy"] for r in results]
layer2_rf_top10_energy=[r["layer2_rf_top10_energy"] for r in results]
unit_sem=lambda key: [np.std(r[key],ddof=1)/np.sqrt(len(r[key])) for r in results]
sparsity_sem=unit_sem("sparsity_units")
layer2_sparsity_sem=unit_sem("layer2_sparsity_units")
filter_similarity_sem=unit_sem("filter_similarity_units")
sta_similarity_sem=unit_sem("sta_similarity_units")
digit_selectivity_sem=unit_sem("digit_selectivity_units")
layer2_digit_selectivity_sem=unit_sem("layer2_digit_selectivity_units")
rf_top10_energy_sem=unit_sem("rf_top10_energy_units")
layer2_rf_top10_energy_sem=unit_sem("layer2_rf_top10_energy_units")
fig,ax=plt.subplots(figsize=(8.5,3.2),dpi=220,facecolor="white")
ax.bar(x-0.22, raw_acc, width=0.22, color="0.55", edgecolor="white", linewidth=.8, alpha=.82, label="raw pixels")
ax.bar(x, sparse_acc, width=0.22, color="#4f6fd5", edgecolor="white", linewidth=.8, alpha=.88, label="layer 1")
ax.bar(x+0.22, layer2_acc, width=0.22, color="#d62728", edgecolor="white", linewidth=.8, alpha=.78, label="layer 1+layer 2")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylim(0,1)
ax.set_ylabel("test accuracy")
ax.set_title("single-layer perceptron on raw MNIST vs Foldiak codes")
fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper center", bbox_to_anchor=(0.5,.98), ncol=3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(rect=[0,0,1,.88])
save_path=os.path.join(save_dir, f"{script_name}_classification_accuracy.png")
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
fig,ax=plt.subplots(figsize=(8.5,3.2),dpi=220,facecolor="white")
ax.bar(x-0.16, sparsity, width=0.32, yerr=sparsity_sem, capsize=2.4, color="#4f6fd5", edgecolor="white", linewidth=.8, alpha=.86, label="layer 1")
ax.bar(x+0.16, layer2_sparsity, width=0.32, yerr=layer2_sparsity_sem, capsize=2.4, color="#d62728", edgecolor="white", linewidth=.8, alpha=.78, label="layer 2")
ax.axhline(p, color="0.45", lw=.9, ls="--", label="target p")
ax.axhline(p2, color="#d62728", lw=.9, ls=":", label="target p2")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("mean activity")
ax.set_title("sparsity of learned MNIST representations")
fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper center", bbox_to_anchor=(0.5,.98), ncol=4)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(rect=[0,0,1,.88])
save_path=os.path.join(save_dir, f"{script_name}_sparsity.png")
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
fig,ax=plt.subplots(figsize=(4.4,3.0),dpi=220,facecolor="white")
ax.axhline(p, color="0.45", lw=.9, ls="--", label="target p")
ax.axhline(p2, color="#d62728", lw=.9, ls=":", label="target p2")
ax.errorbar(x, sparsity, yerr=sparsity_sem, fmt="-o", color="#4f6fd5", lw=1.8, ms=4.8, capsize=2.4, elinewidth=1.0, label="layer 1")
ax.errorbar(x, layer2_sparsity, yerr=layer2_sparsity_sem, fmt="-o", color="#d62728", lw=1.8, ms=4.8, capsize=2.4, elinewidth=1.0, label="layer 2")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylim(0,max(max(sparsity),max(layer2_sparsity),p,p2)*1.18)
ax.set_ylabel("mean activity")
ax.set_title("sparsity of learned MNIST representations")
fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper center", bbox_to_anchor=(0.5,.98), ncol=4, columnspacing=1.0, handlelength=1.6)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(rect=[0,0,1,.88])
save_path=os.path.join(save_dir, f"{script_name}_sparsity_dumbbell.png")
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
fig,axes=plt.subplots(2,2,figsize=(7.2,4.5),dpi=220,facecolor="white")
ax=axes[0,0]
ax.errorbar(x, rf_top10_energy, yerr=rf_top10_energy_sem, fmt="-o", color="#4f6fd5", lw=1.6, ms=3.8, capsize=2.1, elinewidth=.9, label="L1")
ax.errorbar(x, layer2_rf_top10_energy, yerr=layer2_rf_top10_energy_sem, fmt="-o", color="#d62728", lw=1.6, ms=3.8, capsize=2.1, elinewidth=.9, label="L2")
ax.set_title("RF concentration",fontsize=10)
ax.set_ylabel("top 10% energy")
ax.legend(frameon=False,fontsize=7,loc="best")
ax=axes[0,1]
ax.axhline(p,color="0.45",lw=.9,ls="--")
ax.axhline(p2,color="#d62728",lw=.9,ls=":")
ax.errorbar(x, sparsity, yerr=sparsity_sem, fmt="-o", color="#4f6fd5", lw=1.6, ms=3.8, capsize=2.1, elinewidth=.9, label="L1")
ax.errorbar(x, layer2_sparsity, yerr=layer2_sparsity_sem, fmt="-o", color="#d62728", lw=1.6, ms=3.8, capsize=2.1, elinewidth=.9, label="L2")
ax.set_title("sparsity",fontsize=10)
ax.set_ylabel("mean activity")
ax.legend(frameon=False,fontsize=7,loc="lower left")
ax=axes[1,0]
ax.errorbar(x, filter_similarity, yerr=filter_similarity_sem, fmt="-o", color="0.35", lw=1.6, ms=3.8, capsize=2.1, elinewidth=.9, label="weights")
ax.errorbar(x, sta_similarity, yerr=sta_similarity_sem, fmt="-o", color="#9467bd", lw=1.6, ms=3.8, capsize=2.1, elinewidth=.9, label="STA")
ax.set_title("feature redundancy",fontsize=10)
ax.set_ylabel("mean |cosine|")
ax.legend(frameon=False,fontsize=7,loc="best")
ax=axes[1,1]
ax.errorbar(x, digit_selectivity, yerr=digit_selectivity_sem, fmt="-o", color="#4f6fd5", lw=1.6, ms=3.8, capsize=2.1, elinewidth=.9, label="L1")
ax.errorbar(x, layer2_digit_selectivity, yerr=layer2_digit_selectivity_sem, fmt="-o", color="#d62728", lw=1.6, ms=3.8, capsize=2.1, elinewidth=.9, label="L2")
ax.set_title("digit selectivity",fontsize=10)
ax.set_ylabel("max/mean response")
ax.legend(frameon=False,fontsize=7,loc="best")
for ax in axes.reshape(-1):
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
# fig.suptitle("MNIST Foldiak code summary",fontsize=12,y=.99)
plt.tight_layout(rect=[0,0,1,.95],h_pad=.9,w_pad=1.0)
save_path=os.path.join(save_dir, f"{script_name}_summary_metrics.png")
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
for r in results:
    rf1_sta=r["rf1_sta"]
    rf2_sta=r["rf2_sta"]
    qij=r["qij"]
    resolution=r["resolution"]
    number_of_outputs=r["number_of_outputs"]
    layer2_output_size=r["layer2_output_size"]
    ncols=8
    n_show1=min(number_of_outputs,32)
    n_show2=min(layer2_output_size,32)
    nrows1=int(np.ceil(n_show1/ncols))
    nrows2=int(np.ceil(n_show2/ncols))
    fig,axes=plt.subplots(nrows1+nrows2,ncols,figsize=(1.15*ncols,1.15*(nrows1+nrows2)),dpi=220,facecolor="white")
    axes=np.array(axes).reshape(nrows1+nrows2,ncols)
    for i,ax in enumerate(axes[:nrows1].reshape(-1)):
        ax.axis("off")
        if i<n_show1:
            ax.imshow(rf1_sta[i].reshape(resolution,resolution), cmap="gray", interpolation="nearest", vmin=-1, vmax=1)
            ax.set_title(f"L1-{i}",fontsize=7,pad=2)
    for i,ax in enumerate(axes[nrows1:].reshape(-1)):
        ax.axis("off")
        if i<n_show2:
            ax.imshow(rf2_sta[i].reshape(resolution,resolution), cmap="gray", interpolation="nearest", vmin=-1, vmax=1)
            ax.set_title(f"L2-{i}",fontsize=7,pad=2)
    axes[0,0].text(-0.55,0.5,"layer 1 STA",rotation=90,ha="right",va="center",transform=axes[0,0].transAxes,fontsize=9)
    axes[nrows1,0].text(-0.55,0.5,"layer 2 STA",rotation=90,ha="right",va="center",transform=axes[nrows1,0].transAxes,fontsize=9)
    fig.suptitle(f"MNIST activity-triggered RFs | {resolution}px | L1={number_of_outputs}, L2={layer2_output_size}",fontsize=12,y=.99)
    plt.tight_layout()
    save_path=os.path.join(save_dir, f"{script_name}_sta_features_{resolution}px_{number_of_outputs}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    nrows=int(np.ceil(n_show1/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(1.15*ncols,1.15*nrows),dpi=220,facecolor="white")
    axes=np.array(axes).reshape(nrows,ncols)
    vmax=np.max(np.abs(qij[:n_show1]))
    for i,ax in enumerate(axes.reshape(-1)):
        ax.axis("off")
        if i<n_show1:
            ax.imshow(qij[i].reshape(resolution,resolution), cmap="gray", interpolation="nearest", vmin=-vmax, vmax=vmax)
            ax.set_title(f"L1-{i}",fontsize=7,pad=2)
    fig.suptitle(f"MNIST layer 1 weights | {resolution}px | L1={number_of_outputs}",fontsize=12,y=.99)
    plt.tight_layout()
    save_path=os.path.join(save_dir, f"{script_name}_layer1_weights_{resolution}px_{number_of_outputs}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
