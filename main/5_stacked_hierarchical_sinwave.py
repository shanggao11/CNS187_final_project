#%%
import numpy as np
import matplotlib.pyplot as plt
from funs import funs
import os

np.random.seed(0)
plt.rcParams.update({"figure.dpi":140, "axes.spines.top":False, "axes.spines.right":False, "font.size":9})

script_name="5_stacked_hierarchical_sinwave"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
size=(16,16)
num_images=2500
num_classes=4
num_layer1=16
num_layer2=8
alpha1=0.08
beta1=0.02
gamma1=0.02
alpha2=0.08
beta2=0.03
gamma2=0.02
p1=0.15
p2=0.2
dt=0.01
lambda_=10
settling_steps1=250
settling_steps2=250
pretrain_patterns=100

#%%
def wave_component(size, theta, freq, phase=0):
    yy,xx=np.mgrid[0:size[0],0:size[1]]
    x=(xx-size[1]/2)/size[1]
    y=(yy-size[0]/2)/size[0]
    z=np.cos(theta)*x+np.sin(theta)*y
    img=np.sin(2*np.pi*freq*z+phase)
    img=(img-img.min())/(img.max()-img.min()+1e-12)
    return img

def make_templates(size):
    params=[(0,2),(np.pi/2,2),(np.pi/4,2),(-np.pi/4,2),(0,4),(np.pi/2,4),(np.pi/4,4),(-np.pi/4,4)]
    templates=[]
    for theta,freq in params:
        templates.append(wave_component(size, theta, freq))
    return np.array(templates)

def make_hierarchical_sinwave_data(num_images, size):
    images=[]
    labels=[]
    templates=make_templates(size)
    class_components=[[0,5],[1,4],[2,7],[3,6]]
    for i in range(num_images):
        label=np.random.randint(0,len(class_components))
        img=np.zeros(size)
        for component in class_components[label]:
            img+=np.random.uniform(0.8,1.2)*templates[component]
        if np.random.rand()<0.35:
            distractor=np.random.choice([k for k in range(len(templates)) if k not in class_components[label]])
            img+=0.35*templates[distractor]
        img+=0.12*np.random.randn(size[0],size[1])
        img=(img-img.min())/(img.max()-img.min()+1e-12)
        images.append(img)
        labels.append(label)
    return np.array(images),np.array(labels),templates,class_components

def train_layer(allx, number_of_outputs, alpha, beta, gamma, p, settling_steps):
    number_of_inputs=allx.shape[1]
    t_init_i,w_init,q_init=funs.initialization(number_of_outputs, number_of_inputs)
    wij=w_init.copy()
    ti=t_init_i.copy()
    qij=q_init.copy()
    all_y=[]
    for sstep in range(allx.shape[0]):
        xj=allx[sstep:sstep+1].T
        yj_star=np.zeros((number_of_outputs,1))
        yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
        y=funs.binarize(yj_star)
        alpha_now,beta_now,gamma_now=(0.0,0.0,0.1) if sstep<pretrain_patterns else (alpha,beta,gamma)
        qij,wij,ti=funs.update_weights(qij=qij, wij=wij, ti=ti, xj=xj, y=y, alpha=alpha_now, beta=beta_now, gamma=gamma_now, p=p)
        all_y.append(y[:,0])
    return qij,wij,ti,np.array(all_y)

def infer_layer(allx, qij, wij, ti, settling_steps):
    all_y=[]
    for sstep in range(allx.shape[0]):
        xj=allx[sstep:sstep+1].T
        yj_star=np.zeros((qij.shape[0],1))
        yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
        y=funs.binarize(yj_star)
        all_y.append(y[:,0])
    return np.array(all_y)

def class_tuning(all_y, labels, num_classes):
    tuning=np.zeros((all_y.shape[1],num_classes))
    for k in range(num_classes):
        tuning[:,k]=np.mean(all_y[labels==k], axis=0)
    best=np.argmax(tuning, axis=1)
    selectivity=np.max(tuning, axis=1)-np.mean(tuning, axis=1)
    return tuning,best,selectivity

#%%
allimg,labels,templates,class_components=make_hierarchical_sinwave_data(num_images, size)
allx=allimg.reshape(num_images,-1)
q1,w1,t1,y1_train=train_layer(allx, num_layer1, alpha1, beta1, gamma1, p1, settling_steps1)
y1=infer_layer(allx, q1, w1, t1, settling_steps1)
q2,w2,t2,y2_train=train_layer(y1, num_layer2, alpha2, beta2, gamma2, p2, settling_steps2)
y2=infer_layer(y1, q2, w2, t2, settling_steps2)
tuning2,best2,selectivity2=class_tuning(y2, labels, num_classes)
q2_img=q2@q1
 
print("Layer 1 mean activity:", np.mean(y1))
print("Layer 2 mean activity:", np.mean(y2))
for i in range(num_layer2):
    print(f"layer2 neuron {i}: best_class={best2[i]}, selectivity={selectivity2[i]:.3f}, tuning={np.round(tuning2[i],3)}")

#%%
fig,axes=plt.subplots(1,len(templates),figsize=(9,1.7),dpi=220,facecolor="white")
for i,ax in enumerate(axes):
    ax.imshow(templates[i], cmap="gray", interpolation="nearest")
    ax.set_title(f"T{i}", fontsize=8)
    ax.axis("off")
fig.suptitle("base sine-wave components")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_base_components.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
fig,axes=plt.subplots(num_classes,6,figsize=(8,5.2),dpi=220,facecolor="white")
for k in range(num_classes):
    examples=np.where(labels==k)[0][:6]
    for j,idx in enumerate(examples):
        axes[k,j].imshow(allimg[idx], cmap="gray", interpolation="nearest")
        axes[k,j].axis("off")
        if j==0:
            axes[k,j].set_ylabel(f"class {k}\nT{class_components[k][0]}+T{class_components[k][1]}")
fig.suptitle("example hierarchical sine-wave images")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_example_images.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
fig,axes=plt.subplots(1,num_classes,figsize=(8,2.3),dpi=220,facecolor="white")
for k,ax in enumerate(axes):
    ax.imshow(np.mean(allimg[labels==k], axis=0), cmap="gray", interpolation="nearest")
    ax.set_title(f"class {k}")
    ax.axis("off")
fig.suptitle("class mean images")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_class_mean.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
ncols=8
nrows=int(np.ceil(num_layer1/ncols))
fig,axes=plt.subplots(nrows,ncols,figsize=(1.25*ncols,1.25*nrows),dpi=220,facecolor="white")
axes=np.array(axes).reshape(-1)
vmin=np.percentile(q1,2)
vmax=np.percentile(q1,98)
for i,ax in enumerate(axes):
    ax.axis("off")
    if i<num_layer1:
        ax.imshow(q1[i].reshape(size), cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"L1 {i}", fontsize=7)
fig.suptitle("layer 1 learned components")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_layer1_components.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
fig,axes=plt.subplots(1,num_layer2,figsize=(1.35*num_layer2,1.7),dpi=220,facecolor="white")
vmin=np.percentile(q2_img,2)
vmax=np.percentile(q2_img,98)
for i,ax in enumerate(axes):
    ax.imshow(q2_img[i].reshape(size), cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(f"L2 {i}\nclass {best2[i]}", fontsize=7)
    ax.axis("off")
fig.suptitle("layer 2 patterns projected back to image space")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_layer2_projected.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
fig,ax=plt.subplots(figsize=(6.5,3.2),dpi=220,facecolor="white")
im=ax.imshow(tuning2, cmap="viridis", aspect="auto", vmin=0, vmax=max(0.01,np.max(tuning2)))
ax.set_xticks(np.arange(num_classes))
ax.set_xticklabels([f"class {i}" for i in range(num_classes)])
ax.set_yticks(np.arange(num_layer2))
ax.set_yticklabels([f"L2 {i}" for i in range(num_layer2)])
for i in range(num_layer2):
    for j in range(num_classes):
        ax.text(j,i,f"{tuning2[i,j]:.2f}",ha="center",va="center",color="white" if tuning2[i,j]<np.max(tuning2)*0.55 else "black",fontsize=8)
ax.set_title("layer 2 activity by hidden image class")
fig.colorbar(im, ax=ax, pad=.01, label="mean activity")
plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_layer2_class_activity.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
