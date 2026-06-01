#%%
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from funs import funs
import os

script_name="6_mnist_foldiak"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
np.random.seed(0)
torch.manual_seed(0)

resolutions=[14,10]
output_sizes=[64,128]
train_n=3000
test_n=1000
num_batches=2500
settling_steps=120
alpha=0.08
beta=0.02
gamma=0.02
p=0.08
dt=0.01
lambda_=10
pretrain_patterns=100

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
train_x=torch.cat(train_x)[:train_n]
train_y=torch.cat(train_y)[:train_n]
test_x=torch.cat(test_x)[:test_n]
test_y=torch.cat(test_y)[:test_n]

def resize_flat(x, resolution):
    x=F.interpolate(x, size=(resolution,resolution), mode="bilinear", align_corners=False)
    return x.reshape(x.shape[0],-1).numpy()

def train_foldiak(allx, number_of_outputs):
    number_of_inputs=allx.shape[1]
    t_init_i,w_init,q_init=funs.initialization(number_of_outputs, number_of_inputs)
    wij=w_init.copy()
    ti=t_init_i.copy()
    qij=q_init.copy()
    activity=[]
    for sstep in range(num_batches):
        xj=allx[sstep:sstep+1].T
        yj_star=np.zeros((number_of_outputs,1))
        yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
        y=funs.binarize(yj_star)
        alpha_now,beta_now,gamma_now=(0.0,0.0,0.1) if sstep<pretrain_patterns else (alpha,beta,gamma)
        qij,wij,ti=funs.update_weights(qij=qij, wij=wij, ti=ti, xj=xj, y=y, alpha=alpha_now, beta=beta_now, gamma=gamma_now, p=p)
        activity.append(np.mean(y))
    return qij,wij,ti,np.array(activity)

def encode_foldiak(allx, qij, wij, ti):
    all_y=[]
    for sstep in range(allx.shape[0]):
        xj=allx[sstep:sstep+1].T
        yj_star=np.zeros((qij.shape[0],1))
        yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
        all_y.append(funs.binarize(yj_star)[:,0])
    return np.array(all_y)

def perceptron_accuracy(x_train, y_train, x_test, y_test, epochs=25, lr=0.1):
    x_train=torch.tensor(x_train, dtype=torch.float32)
    x_test=torch.tensor(x_test, dtype=torch.float32)
    y_train=torch.tensor(y_train, dtype=torch.long)
    y_test=torch.tensor(y_test, dtype=torch.long)
    model=nn.Linear(x_train.shape[1],10)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        opt.zero_grad()
        loss=F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc=(model(x_train).argmax(1)==y_train).float().mean().item()
        test_acc=(model(x_test).argmax(1)==y_test).float().mean().item()
    return train_acc,test_acc

results=[]
for resolution in resolutions:
    x_train=resize_flat(train_x, resolution)
    x_test=resize_flat(test_x, resolution)
    raw_train_acc,raw_test_acc=perceptron_accuracy(x_train, train_y.numpy(), x_test, test_y.numpy())
    for number_of_outputs in output_sizes:
        qij,wij,ti,activity=train_foldiak(x_train[:num_batches], number_of_outputs)
        z_train=encode_foldiak(x_train, qij, wij, ti)
        z_test=encode_foldiak(x_test, qij, wij, ti)
        sparse_train_acc,sparse_test_acc=perceptron_accuracy(z_train, train_y.numpy(), z_test, test_y.numpy())
        result={"resolution":resolution, "number_of_outputs":number_of_outputs, "raw_test_acc":raw_test_acc, "sparse_test_acc":sparse_test_acc, "sparsity":np.mean(z_train), "activity":activity, "qij":qij}
        results.append(result)
        print(f"res={resolution}, outputs={number_of_outputs}, raw_acc={raw_test_acc:.3f}, sparse_acc={sparse_test_acc:.3f}, sparse_activity={np.mean(z_train):.3f}")

#%%
names=[f"{r['resolution']}px/{r['number_of_outputs']}" for r in results]
x=np.arange(len(results))
raw_acc=[r["raw_test_acc"] for r in results]
sparse_acc=[r["sparse_test_acc"] for r in results]
sparsity=[r["sparsity"] for r in results]

fig,ax=plt.subplots(figsize=(8.5,3.2),dpi=220,facecolor="white")
ax.bar(x-0.18, raw_acc, width=0.36, color="0.55", edgecolor="white", linewidth=.8, alpha=.82, label="raw pixels")
ax.bar(x+0.18, sparse_acc, width=0.36, color="#4f6fd5", edgecolor="white", linewidth=.8, alpha=.88, label="Foldiak sparse code")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylim(0,1)
ax.set_ylabel("test accuracy")
ax.set_title("single-layer perceptron on raw MNIST vs sparse code")
ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y",color="0.88",lw=.8)
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_classification_accuracy.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
fig,ax=plt.subplots(figsize=(8.5,3.2),dpi=220,facecolor="white")
ax.bar(x, sparsity, width=0.45, color="#d62728", edgecolor="white", linewidth=.8, alpha=.82)
ax.axhline(p, color="0.45", lw=.9, ls="--", label="target p")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("mean activity")
ax.set_title("sparsity of learned MNIST representation")
ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y",color="0.88",lw=.8)
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_sparsity.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
for r in results:
    qij=r["qij"]
    resolution=r["resolution"]
    number_of_outputs=r["number_of_outputs"]
    ncols=8
    nrows=int(np.ceil(min(number_of_outputs,32)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(1.15*ncols,1.15*nrows),dpi=220,facecolor="white")
    axes=np.array(axes).reshape(-1)
    vmin=np.percentile(qij,2)
    vmax=np.percentile(qij,98)
    for i,ax in enumerate(axes):
        ax.axis("off")
        ax.imshow(qij[i].reshape(resolution,resolution), cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{i}",fontsize=7,pad=2)
    fig.suptitle(f"MNIST learned features | {resolution}px | outputs={number_of_outputs}",fontsize=12,y=.99)
    plt.tight_layout()
    save_path=os.path.join(save_dir, f"{script_name}_features_{resolution}px_{number_of_outputs}.png")
    if not os.path.exists(save_path):
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
