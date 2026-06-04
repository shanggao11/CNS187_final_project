#%%
import numpy as np
import matplotlib.pyplot as plt
from funs import funs
import os
from joblib import Parallel, delayed

np.random.seed(0)
plt.rcParams.update({"figure.dpi":140, "axes.spines.top":False, "axes.spines.right":False, "font.size":9})
script_name="8_foldiak_overcomplete_natural"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)

dataload_path="./data/bruno_sparse/IMAGES_Vanhateren.npy"
xdim=16
ydim=16
patch_dim=xdim*ydim
# output_sizes=[256,2048]
# output_sizes=[256,1024]
# summary_output_sizes=[256,1024]
output_sizes=[2048]
summary_output_sizes=[2048]
num_test=1000
settling_steps=300
batch_size=1

base_run_tag="bruno_match_long"
dt=0.01
l2norm=True
input_mode="zca" # zca/onoff

########### param 1
# learning_rule="foldiak" # soft_oja/foldiak
# num_train=30000
# alpha=0.03
# beta=0.005  
# gamma=0.01
# p=0.03 #todo, 0.03 default
# lambda_=10
########## param 2, paper's param
learning_rule="soft_oja" # soft_oja/foldiak
num_train=30000
beta = 3.0         
alpha = 2.4      
gamma = 1.5
p=0.005
lambda_=200
##############
pretrain_patterns=300
n_jobs=-1
print_every=1000
if l2norm:
    l2normname="_l2norm"
else:
    l2normname=""
D=patch_dim
param_tag=f"{input_mode}_{learning_rule}_p{p:g}_a{alpha:g}_b{beta:g}_g{gamma:g}_l{lambda_:g}_bs{batch_size}_st{settling_steps}{l2normname}"
run_tag=f"{base_run_tag}_{param_tag}"

def condition_dir(number_of_outputs):
    path=os.path.join(save_dir, f"N{number_of_outputs}_{param_tag}")
    os.makedirs(path, exist_ok=True)
    return path

def comparison_dir(results):
    names="_".join([f"N{r['number_of_outputs']}" for r in results])
    path=os.path.join(save_dir, f"{names}_{param_tag}")
    os.makedirs(path, exist_ok=True)
    return path

def is_finite(*arrays):
    return all(np.all(np.isfinite(a)) for a in arrays)

def preprocess_patches(train_patches_raw, test_patches_raw):
    if input_mode=="zca":
        train_patches,test_patches,W,mean_x=funs.zca_whiten(train_patches_raw, test_patches_raw)
        return train_patches,test_patches,W,mean_x
    if input_mode=="onoff":
        train_patches=np.maximum(train_patches_raw,0)
        test_patches=np.maximum(test_patches_raw,0)
        train_patches[1::2]=np.maximum(-train_patches_raw[1::2],0)
        test_patches[1::2]=np.maximum(-test_patches_raw[1::2],0)
        return train_patches,test_patches,np.array([]),np.array([])
    raise ValueError("input_mode should be zca or onoff")

def show_filter(q):
    return q.reshape(xdim,ydim)

def update_soft_oja(qij, wij, ti, xj, y, alpha, beta, gamma, p):
    now_batch_size=xj.shape[1]
    yiyj=(y@y.T)/now_batch_size
    yi_mean=np.mean(y, axis=1, keepdims=True)
    yixj=(y@xj.T)/now_batch_size
    yi2_mean=np.mean(y**2, axis=1, keepdims=True)
    wij=wij-alpha*(yiyj-p**2)
    np.fill_diagonal(wij,0)
    wij=np.minimum(wij,0)
    ti=ti+gamma*(yi_mean-p)
    qij=qij+beta*(yixj-yi2_mean*qij) # only difference is we have a y^2 term. 
    return qij,wij,ti

def train_foldiak(allx, number_of_outputs, settling_steps):
    number_of_inputs=allx.shape[1]
    t_init_i,w_init,q_init=funs.initialization(number_of_outputs, number_of_inputs)
    q_init=np.random.randn(number_of_outputs, number_of_inputs)
    q_init=q_init/(np.linalg.norm(q_init, axis=1, keepdims=True)+1e-12)
    wij=w_init.copy()
    ti=t_init_i.copy()
    qij=q_init.copy()
    activity=[]
    num_updates=int(np.ceil(allx.shape[0]/batch_size))
    for sstep in range(num_updates):
        if sstep%print_every==0:
            print(f"training N={number_of_outputs}: {sstep}/{num_updates}")
        xj=allx[sstep*batch_size:(sstep+1)*batch_size].T
        yj_star=np.zeros((number_of_outputs,xj.shape[1]))
        yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
        if not is_finite(yj_star):
            raise ValueError(f"NaN or inf in yj_star at step {sstep}, try smaller beta/lambda/alpha/gamma")
        y=funs.binarize(yj_star) if learning_rule=="foldiak" else yj_star
        alpha_now,beta_now,gamma_now=(0.0,0.0,0.1) if sstep<pretrain_patterns else (alpha,beta,gamma)
        if learning_rule=="foldiak":
            qij,wij,ti=funs.update_weights(qij=qij, wij=wij, ti=ti, xj=xj, y=y, alpha=alpha_now, beta=beta_now, gamma=gamma_now, p=p)
        if learning_rule=="soft_oja":
            qij,wij,ti=update_soft_oja(qij, wij, ti, xj, y, alpha_now, beta_now, gamma_now, p)
        if not is_finite(qij,wij,ti):
            raise ValueError(f"NaN or inf in weights at step {sstep}, try smaller beta/lambda/alpha/gamma")
        if l2norm:
            qij=qij/(np.linalg.norm(qij, axis=1, keepdims=True)+1e-12) #todo , e.g qij is (16,64), 
        else:
            pass
        activity.append(np.mean(y))
    print(f"training N={number_of_outputs}: done")
    return qij,wij,ti,np.array(activity)

def encode_one(x, qij, wij, ti, settling_steps):
    xj=x.reshape(-1,1)
    yj_star=np.zeros((qij.shape[0],1))
    yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
    y=funs.binarize(yj_star) if learning_rule=="foldiak" else yj_star
    return y[:,0]

def encode_foldiak(allx, qij, wij, ti, settling_steps):
    print(f"encoding N={qij.shape[0]}: {allx.shape[0]} samples")
    if Parallel is not None:
        all_y=np.array(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(encode_one)(allx[sstep], qij, wij, ti, settling_steps) for sstep in range(allx.shape[0])))
        print(f"encoding N={qij.shape[0]}: done")
        return all_y
    all_y=[]
    for sstep in range(allx.shape[0]):
        if sstep%print_every==0:
            print(f"encoding N={qij.shape[0]}: {sstep}/{allx.shape[0]}")
        all_y.append(encode_one(allx[sstep], qij, wij, ti, settling_steps))
    print(f"encoding N={qij.shape[0]}: done")
    return np.array(all_y)

#%%
results=[]
for number_of_outputs in output_sizes:
    print(f"start condition N={number_of_outputs}")
    one_save_dir=condition_dir(number_of_outputs)
    patch_cache=os.path.join(one_save_dir, f"patches_{input_mode}_train{num_train}_test{num_test}_size{xdim}.npz")
    if os.path.exists(patch_cache):
        cache=np.load(patch_cache)
        train_patches_raw,test_patches_raw,train_patches,test_patches,W,mean_x=cache["train_patches_raw"],cache["test_patches_raw"],cache["train_patches"],cache["test_patches"],cache["W"],cache["mean_x"]
    else:
        vm=np.load(dataload_path)
        train_patches_raw=funs.sample_natural_patches(vm, num_train, xdim, ydim)
        test_patches_raw=funs.sample_natural_patches(vm, num_test, xdim, ydim)
        train_patches,test_patches,W,mean_x=preprocess_patches(train_patches_raw, test_patches_raw)
        np.savez_compressed(patch_cache, train_patches_raw=train_patches_raw, test_patches_raw=test_patches_raw, train_patches=train_patches, test_patches=test_patches, W=W, mean_x=mean_x)
    fig,axes=plt.subplots(2,8,figsize=(8.5,2.4),dpi=220,facecolor="white")
    axes=np.array(axes).reshape(-1)
    for i,ax in enumerate(axes):
        ax.imshow(train_patches_raw[i].reshape(xdim,ydim), cmap="gray", interpolation="nearest")
        ax.axis("off")
    fig.suptitle("example natural image patches before whitening")
    plt.tight_layout()
    funs.save_fig(fig, one_save_dir, script_name, run_tag, "example_patches")
    # plt.show()
    result_cache=os.path.join(one_save_dir, f"result_train{num_train}_batch{batch_size}_test{num_test}_settle{settling_steps}.npz")
    if os.path.exists(result_cache):
        cache=np.load(result_cache)
        qij,wij,ti,activity,y_test=cache["qij"],cache["wij"],cache["ti"],cache["activity"],cache["y_test"]
        sparsity,corr,similarity,diversity=float(cache["sparsity"]),float(cache["corr"]),float(cache["similarity"]),float(cache["diversity"])
        cache_good=is_finite(qij,wij,ti,activity,y_test,np.array([sparsity,corr,similarity,diversity]))
        if not cache_good:
            print(f"cached result has NaN, rerunning N={number_of_outputs}")
    else:
        cache_good=False
    if not cache_good:
        qij,wij,ti,activity=train_foldiak(train_patches[:num_train], number_of_outputs, settling_steps)
        y_test=encode_foldiak(test_patches, qij, wij, ti, settling_steps)
        sparsity=np.mean(y_test)
        corr=funs.output_corr(y_test)
        similarity=funs.filter_similarity(qij)
        diversity=1-similarity
        if not is_finite(qij,wij,ti,activity,y_test,np.array([sparsity,corr,similarity,diversity])):
            raise ValueError(f"NaN or inf in final result for N={number_of_outputs}, try smaller beta/lambda/alpha/gamma")
        np.savez_compressed(result_cache, qij=qij, wij=wij, ti=ti, activity=activity, y_test=y_test, sparsity=sparsity, corr=corr, similarity=similarity, diversity=diversity)
    completeness=number_of_outputs/D
    results.append({"number_of_outputs":number_of_outputs, "completeness":completeness, "qij":qij, "activity":activity, "y_test":y_test, "sparsity":sparsity, "corr":corr, "similarity":similarity, "diversity":diversity})
    print(f"N={number_of_outputs}, N/D={completeness:.2f}, sparsity={sparsity:.3f}, output_corr={corr:.3f}, filter_similarity={similarity:.3f}, diversity={diversity:.3f}")

for number_of_outputs in summary_output_sizes:
    one_save_dir=condition_dir(number_of_outputs)
    result_cache=os.path.join(one_save_dir, f"result_train{num_train}_batch{batch_size}_test{num_test}_settle{settling_steps}.npz")
    if os.path.exists(result_cache) and number_of_outputs not in [r["number_of_outputs"] for r in results]:
        cache=np.load(result_cache)
        qij,activity,y_test=cache["qij"],cache["activity"],cache["y_test"]
        sparsity,corr,similarity,diversity=float(cache["sparsity"]),float(cache["corr"]),float(cache["similarity"]),float(cache["diversity"])
        completeness=number_of_outputs/D
        results.append({"number_of_outputs":number_of_outputs, "completeness":completeness, "qij":qij, "activity":activity, "y_test":y_test, "sparsity":sparsity, "corr":corr, "similarity":similarity, "diversity":diversity})
results=sorted(results, key=lambda r:r["number_of_outputs"])
plot_dir=condition_dir(results[0]["number_of_outputs"]) if len(results)==1 else comparison_dir(results)

#%%
names=[f"{r['number_of_outputs']}\n({r['completeness']:.1f}x)" for r in results]
x=np.arange(len(results))
sparsity=[r["sparsity"] for r in results]
corr=[r["corr"] for r in results]
similarity=[r["similarity"] for r in results]
diversity=[r["diversity"] for r in results]

fig,ax=plt.subplots(figsize=(8.5,3.2),dpi=220,facecolor="white")
ax.bar(x-0.3, sparsity, width=0.2, color="#4f6fd5", edgecolor="white", linewidth=.8, alpha=.88, label="sparsity")
ax.bar(x-0.1, corr, width=0.2, color="0.55", edgecolor="white", linewidth=.8, alpha=.82, label="output corr")
ax.bar(x+0.1, similarity, width=0.2, color="#d62728", edgecolor="white", linewidth=.8, alpha=.78, label="filter similarity")
ax.bar(x+0.3, diversity, width=0.2, color="#f58518", edgecolor="white", linewidth=.8, alpha=.82, label="filter diversity")
ax.axhline(p, color="0.35", lw=.9, ls="--", label="target p")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0,1)
ax.set_xlabel("number of output units N")
ax.set_ylabel("metric")
ax.set_title("Foldiak overcompleteness on natural image patches")
ax.legend(frameon=False, ncols=3, fontsize=8)
ax.grid(axis="y", color="0.88", lw=.8)
plt.tight_layout()
funs.save_fig(fig, plot_dir, script_name, run_tag, "metrics_summary")
# plt.show()

#%%
fig,ax=plt.subplots(figsize=(7.2,3.2),dpi=220,facecolor="white")
ax.plot([r["completeness"] for r in results], sparsity, marker="o", color="#4f6fd5", lw=2, label="sparsity")
ax.plot([r["completeness"] for r in results], corr, marker="s", color="0.35", lw=2, label="output corr")
ax.plot([r["completeness"] for r in results], similarity, marker="^", color="#d62728", lw=2, label="filter similarity")
ax.set_xlabel("overcompleteness N/D")
ax.set_ylabel("metric")
ax.set_ylim(0,1)
ax.set_title("effect of overcompleteness")
ax.legend(frameon=False)
ax.grid(color="0.9", lw=.8)
plt.tight_layout()
funs.save_fig(fig, plot_dir, script_name, run_tag, "metrics_curve")
# plt.show()

#%%
for r in results:
    qij=r["qij"]
    qplot=qij-np.mean(qij, axis=1, keepdims=True)
    number_of_outputs=r["number_of_outputs"]
    n_show=min(64,number_of_outputs)
    ncols=8
    nrows=int(np.ceil(n_show/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(1.15*ncols,1.15*nrows),dpi=220,facecolor="white")
    axes=np.array(axes).reshape(-1)
    v=np.percentile(np.abs(qplot[:n_show]),98)
    for i,ax in enumerate(axes):
        ax.imshow(qplot[i].reshape(xdim,ydim), cmap="gray", interpolation="nearest", vmin=-v, vmax=v)
        ax.set_title(f"{i}", fontsize=7, pad=2)
        ax.axis("off")
    fig.suptitle(f"Foldiak filters on natural patches | N={number_of_outputs}, N/D={r['completeness']:.1f}x", fontsize=12, y=.99)
    plt.tight_layout()
    funs.save_fig(fig, condition_dir(number_of_outputs), script_name, run_tag, f"filters_N{number_of_outputs}")
    # plt.show()

#%%
metric_mat=np.array([sparsity,corr,similarity,diversity])
metric_names=["sparsity","output corr","filter similarity","filter diversity"]
fig,ax=plt.subplots(figsize=(7.5,3.0),dpi=220,facecolor="white")
im=ax.imshow(metric_mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_yticks(np.arange(len(metric_names)))
ax.set_yticklabels(metric_names)
for i in range(metric_mat.shape[0]):
    for j in range(metric_mat.shape[1]):
        ax.text(j,i,f"{metric_mat[i,j]:.2f}",ha="center",va="center",color="white" if metric_mat[i,j]<0.55 else "black",fontsize=8)
ax.set_title("summary metrics")
fig.colorbar(im, ax=ax, pad=.01)
plt.tight_layout()
funs.save_fig(fig, plot_dir, script_name, run_tag, "metrics_heatmap")
# plt.show()
