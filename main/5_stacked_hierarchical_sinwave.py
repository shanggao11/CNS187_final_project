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
num_layer1=12
num_layer2=4
alpha1=0.08
beta1=0.02
gamma1=0.02
alpha2=0.08
beta2=0.03
gamma2=0.02
p1=0.18
p2=0.25
dt=0.01
lambda_=10
settling_steps1=220
settling_steps2=220
pretrain_patterns=100

#%%
def make_grating(size, theta, freq, phase=0):
    yy,xx=np.mgrid[0:size[0],0:size[1]]
    x=(xx-size[1]/2)/size[1]
    y=(yy-size[0]/2)/size[0]
    z=np.cos(theta)*x+np.sin(theta)*y
    img=np.sin(2*np.pi*freq*z+phase)
    img=(img-img.min())/(img.max()-img.min()+1e-12)
    return img

def make_three_gratings(size):
    return np.array([make_grating(size, 0, 3), make_grating(size, np.pi/3, 3), make_grating(size, -np.pi/3, 3)])

def make_fake_data(num_images, size):
    gratings=make_three_gratings(size)
    patterns=[[0,1],[1,2],[0,2]] # each fake image is a higher-order pattern made from two simple gratings
    images,labels=[],[]
    for i in range(num_images):
        label=np.random.randint(len(patterns))
        img=np.zeros(size)
        for k in patterns[label]:
            img+=np.random.uniform(0.8,1.2)*gratings[k]
        img+=0.10*np.random.randn(size[0],size[1])
        img=(img-img.min())/(img.max()-img.min()+1e-12)
        images.append(img); labels.append(label)
    return np.array(images),np.array(labels),gratings,patterns

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
        all_y.append(funs.binarize(yj_star)[:,0])
    return np.array(all_y)

def save_show(fig, name):
    plt.tight_layout()
    save_path=os.path.join(save_dir, f"{script_name}_{name}.png")
    if not os.path.exists(save_path):
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_filters(qij, title, name, ncols=6):
    nrows=int(np.ceil(qij.shape[0]/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(1.3*ncols,1.3*nrows),dpi=220,facecolor="white")
    axes=np.array(axes).reshape(-1)
    vmin,vmax=np.percentile(qij,2),np.percentile(qij,98)
    for i,ax in enumerate(axes):
        ax.axis("off")
        if i<qij.shape[0]:
            ax.imshow(qij[i].reshape(size), cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(f"{i}", fontsize=8)
    fig.suptitle(title)
    save_show(fig, name)

#%%
allimg,labels,gratings,patterns=make_fake_data(num_images, size)
allx=allimg.reshape(num_images,-1)
q1,w1,t1,y1_train=train_layer(allx, num_layer1, alpha1, beta1, gamma1, p1, settling_steps1)
y1=infer_layer(allx, q1, w1, t1, settling_steps1)
q2,w2,t2,y2_train=train_layer(y1, num_layer2, alpha2, beta2, gamma2, p2, settling_steps2)
y2=infer_layer(y1, q2, w2, t2, settling_steps2)
q2_image=q2@q1 # layer 2 weights are over layer 1 neurons, so multiply by q1 to see the image pattern

print("Layer 1 mean activity:", np.mean(y1))
print("Layer 2 mean activity:", np.mean(y2))

#%%
fig,axes=plt.subplots(1,3,figsize=(4.5,1.6),dpi=220,facecolor="white")
for i,ax in enumerate(axes):
    ax.imshow(gratings[i], cmap="gray", interpolation="nearest")
    ax.set_title(f"grating {i}", fontsize=8)
    ax.axis("off")
fig.suptitle("three simple grating components")
save_show(fig, "three_gratings")

#%%
fig,axes=plt.subplots(3,5,figsize=(6.5,3.8),dpi=220,facecolor="white")
for k in range(3):
    examples=np.where(labels==k)[0][:5]
    for j,idx in enumerate(examples):
        axes[k,j].imshow(allimg[idx], cmap="gray", interpolation="nearest")
        axes[k,j].axis("off")
        if j==0:
            axes[k,j].set_ylabel(f"G{patterns[k][0]}+G{patterns[k][1]}")
fig.suptitle("fake images made by combining two gratings")
save_show(fig, "fake_images")

#%%
plot_filters(q1, "layer 1 learned weights", "layer1_weights", ncols=6)
plot_filters(q2_image, "layer 2 learned weights shown in image space", "layer2_weights", ncols=4)
