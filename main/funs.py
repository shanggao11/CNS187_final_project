import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class funs:
    def prepare_mnist_data(batch_size=64):
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        test_data = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def resize_flat(x, resolution):
        import torch.nn.functional as F
        x=F.interpolate(x, size=(resolution,resolution), mode="bilinear", align_corners=False)
        return x.reshape(x.shape[0],-1).numpy()

    def train_foldiak(allx, number_of_outputs, p, alpha, beta, gamma, lambda_=10, dt=0.01, settling_steps=120, num_batches=None, pretrain_patterns=100, batch_size=1, checkpoints=None,gammatuned=False):
        num_batches=int(np.ceil(allx.shape[0]/batch_size)) if num_batches is None else num_batches
        pretrain_updates=int(np.ceil(pretrain_patterns/batch_size))
        checkpoints=[] if checkpoints is None else checkpoints
        number_of_inputs=allx.shape[1]
        t_init_i,w_init,q_init=funs.initialization(number_of_outputs, number_of_inputs)
        wij=w_init.copy()
        ti=t_init_i.copy()
        qij=q_init.copy()
        activity=[]
        qij_checkpoints={}
        for sstep in range(num_batches):
            xj=allx[sstep*batch_size:(sstep+1)*batch_size].T
            yj_star=np.zeros((number_of_outputs,xj.shape[1]))
            yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
            y=funs.binarize(yj_star)
            if gammatuned:
                alpha_now,beta_now,gamma_now=(0.0,0.0,gamma) if sstep<pretrain_updates else (alpha,beta,gamma) 
            else:
                alpha_now,beta_now,gamma_now=(0.0,0.0,0.1) if sstep<pretrain_updates else (alpha,beta,gamma) # todo, gamma before or after warm up? 
            qij,wij,ti=funs.update_weights(qij=qij, wij=wij, ti=ti, xj=xj, y=y, alpha=alpha_now, beta=beta_now, gamma=gamma_now, p=p)
            activity.append(np.mean(y))
            if sstep in checkpoints:
                qij_checkpoints[sstep]=qij.copy()
        if len (checkpoints)>0:
            return (qij,wij,ti,np.array(activity),qij_checkpoints)
        else:
            return (qij,wij,ti,np.array(activity))


    def encode_foldiak(allx, qij, wij, ti, lambda_=10, dt=0.01, settling_steps=120):
        all_y=[]
        for sstep in range(allx.shape[0]):
            xj=allx[sstep:sstep+1].T
            yj_star=np.zeros((qij.shape[0],1))
            yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
            all_y.append(funs.binarize(yj_star)[:,0])
        return np.array(all_y)

    def activity_rf(allx, all_y):
        rf=(all_y.T@allx)/(np.sum(all_y, axis=0, keepdims=True).T+1e-12)
        rf=rf-np.mean(rf, axis=1, keepdims=True) # demean
        rf=rf/(np.max(np.abs(rf), axis=1, keepdims=True)+1e-12) # scale to -1 or 1
        return rf

    def perceptron_accuracy(x_train, y_train, x_test, y_test, epochs=25, lr=0.1):
        import torch
        from torch import nn
        import torch.nn.functional as F
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

    def mean_abs_offdiag_cosine(a):
        a=np.asarray(a); 
        a=a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-12); 
        c=np.abs(a@a.T)
        return (np.sum(c)-np.trace(c))/(c.size-len(c))
    def digit_selectivity(z,y):
        y=np.asarray(y)
        responses=np.vstack([np.mean(z[y==d],axis=0) for d in range(10)])
        return np.mean(np.max(responses,axis=0)/(np.mean(responses,axis=0)+1e-12))
    def top_energy_fraction(rf, frac=.1, average=True):
        e=np.sort(np.asarray(rf).reshape(rf.shape[0],-1)**2,axis=1)
        k=max(1,int(np.ceil(e.shape[1]*frac))) # 10 
        out=np.sum(e[:,-k:],axis=1)/(np.sum(e,axis=1)+1e-12)
        return np.mean(out) if average else out
    def cosine_stats(rf):
        q=np.asarray(rf).reshape(rf.shape[0],-1); q=q-np.mean(q,axis=1,keepdims=True); q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12)
        sim=np.abs(q@q.T); mask=~np.eye(sim.shape[0],dtype=bool)
        return np.mean(sim[mask]),np.percentile(np.max(sim-np.eye(sim.shape[0]),axis=1),90)

    def update_soft_oja(qij, wij, ti, xj, y, alpha, beta, gamma, p):
        batch_size=xj.shape[1]
        yiyj=(y@y.T)/batch_size
        yi_mean=np.mean(y, axis=1, keepdims=True)
        yixj=(y@xj.T)/batch_size
        yi2_mean=np.mean(y**2, axis=1, keepdims=True)
        wij=wij-alpha*(yiyj-p**2)
        np.fill_diagonal(wij,0)
        wij=np.minimum(wij,0)
        ti=ti+gamma*(yi_mean-p)
        qij=qij+beta*(yixj-yi2_mean*qij)
        return qij,wij,ti
    def train_soft_oja_foldiak(allx, number_of_outputs, p, alpha, beta, gamma, lambda_=10, dt=0.01, settling_steps=120, num_batches=None, pretrain_patterns=100):
        num_batches=allx.shape[0] if num_batches is None else num_batches
        number_of_inputs=allx.shape[1]
        t_init_i,w_init,q_init=funs.initialization(number_of_outputs, number_of_inputs)
        q_init=np.random.randn(number_of_outputs, number_of_inputs)
        q_init=q_init/(np.linalg.norm(q_init, axis=1, keepdims=True)+1e-12)
        wij=w_init.copy()
        ti=t_init_i.copy()
        qij=q_init.copy()
        activity=[]
        for sstep in range(num_batches):
            xj=allx[sstep:sstep+1].T
            yj_star=np.zeros((number_of_outputs,1))
            yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
            y=yj_star
            alpha_now,beta_now,gamma_now=(0.0,0.0,0.1) if sstep<pretrain_patterns else (alpha,beta,gamma)
            qij,wij,ti=funs.update_soft_oja(qij, wij, ti, xj, y, alpha_now, beta_now, gamma_now, p)
            activity.append(np.mean(y))
        return qij,wij,ti,np.array(activity)
    def encode_soft_foldiak(allx, qij, wij, ti, lambda_=10, dt=0.01, settling_steps=120):
        all_y=[]
        for sstep in range(allx.shape[0]):
            xj=allx[sstep:sstep+1].T
            yj_star=np.zeros((qij.shape[0],1))
            yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
            all_y.append(yj_star[:,0])
        return np.array(all_y)

    def update_weights(qij, wij, ti, xj, y, alpha, beta, gamma, p, q_decay="mean", dtype=None):
        batch_size = xj.shape[1]
        yiyj= (y @ y.T) / batch_size
        yi_mean = np.mean(y, axis=1, keepdims=True)
        yixj = (y @ xj.T) / batch_size
        yi_decay = np.mean(y**2, axis=1, keepdims=True) if q_decay=="second_moment" else yi_mean
        delta_w = -alpha * (yiyj - p**2)
        delta_ti = gamma * (yi_mean - p)
        # delta_qij = beta * y * (xj.T - qij)
        delta_qij = beta * (yixj - yi_decay * qij)
        wij = wij + delta_w
        np.fill_diagonal(wij, 0)
        wij = np.minimum(wij, 0)
        ti = ti + delta_ti
        qij = qij + delta_qij
        if dtype is not None:
            qij,wij,ti=qij.astype(dtype),wij.astype(dtype),ti.astype(dtype)
        return qij, wij, ti
    def settling_y(qij, wij, ti, xj, yj_star=None, lambda_=10, dt=0.01, settling_steps=300, clip_sigmoid_input=None, dtype=None):
        if yj_star is None:
            yj_star=np.zeros((qij.shape[0], xj.shape[1]), dtype=dtype)
        feedforward_i = qij @ xj - ti
        for _ in range(settling_steps):
            dystar_i= (funs.sigmoid(feedforward_i + wij@yj_star, lambda_, clip_sigmoid_input) - yj_star) * dt
            yj_star += dystar_i
        return yj_star
    def initialization(number_of_outputs, number_of_inputs, threshold_init="random", dtype=None, return_order="twq"):
        t_init_i=np.zeros((number_of_outputs,1)) if threshold_init=="zero" else (np.random.rand(number_of_outputs) * 0.01 - 0.005).reshape(number_of_outputs,1) # small random values for initial thresholds
        w_init=np.zeros((number_of_outputs, number_of_outputs))
        q_init=np.random.rand(number_of_outputs, number_of_inputs)
        q_init = q_init /np.linalg.norm(q_init, axis=1, keepdims=True) # every row is output neuron, every column is input to that output neuorn. 
        if dtype is not None:
            t_init_i,w_init,q_init=t_init_i.astype(dtype),w_init.astype(dtype),q_init.astype(dtype)
        return (q_init,w_init,t_init_i) if return_order=="qwt" else (t_init_i,w_init,q_init)

    def sigmoid(x,lambda_=10, clip_sigmoid_input=None):
        z=lambda_*x if clip_sigmoid_input is None else np.clip(lambda_*x, -clip_sigmoid_input, clip_sigmoid_input)
        y=1/(1+np.exp(-z))
        return y
    def is_finite(*arrays):
        return all(np.all(np.isfinite(a)) for a in arrays)
    def onoff_encode(patches_raw, dtype=None):
        x_on=np.maximum(patches_raw,0)
        x_off=np.maximum(-patches_raw,0)
        x=np.concatenate([x_on,x_off], axis=1)
        return x.astype(dtype) if dtype is not None else x
    def component_from_onoff(qij, patch_dim, normalize="maxabs"):
        q_on=qij[:,:patch_dim]
        q_off=qij[:,patch_dim:]
        comp=q_on-q_off
        comp=comp-np.mean(comp, axis=1, keepdims=True)
        if normalize=="maxabs":
            comp=comp/(np.max(np.abs(comp), axis=1, keepdims=True)+1e-12)
        return comp
    def binarize(ystar, threshold=0.5):
        return (ystar > threshold).astype(float)
    def make_batches_line_pattern(num_batches, size=(8, 8), p_line=1/8, thickness=1,background=0):
        batches = []
        for _ in range(num_batches):
            img = funs.make_one_line_pattern(size=size, p_line=p_line, thickness=thickness, background=background)
            batches.append(img)
        allimg=np.array(batches)
        return allimg
    def make_one_line_pattern(size=(8, 8), p_line=1/8, thickness=1,background=0):
        if background==0:
            img = np.zeros(size, dtype=float)
            fill_value=1
        else:
            img = np.ones(size, dtype=float)
            fill_value=0
        # horizontal lines
        for r in range(size[0]):
            if np.random.rand() < p_line:
                img[max(0, r-thickness+1):r+1, :] = fill_value
        # vertical lines
        for c in range(size[1]):
            if np.random.rand() < p_line:
                img[:, max(0, c-thickness+1):c+1] = fill_value
        return img
    def make_line_masks(size):
        # make line with perfect line, from top to bottom rows, then left to right columns, return number_of_outputs x number_of_inputs 
        masks=[]
        for r in range(size[0]):
            img=np.zeros(size); img[r,:]=1; masks.append(img.reshape(-1))
        for c in range(size[1]):
            img=np.zeros(size); img[:,c]=1; masks.append(img.reshape(-1))
        masks=np.array(masks); masks=masks/(np.linalg.norm(masks, axis=1, keepdims=True)+1e-12)
        return masks
    def line_score(qij, size):
        # qij: number_of_outputs x number_of_inputs
        q=qij/(np.linalg.norm(qij, axis=1, keepdims=True)+1e-12)
        sim=q@funs.make_line_masks(size).T # note: making line with perfect line, from top to bottom rows, then left to right columns # shape number_of_outputs x number_of_outputs
        # -> 16x64 @ 64x16 -> 16x16 for 8x8 input, 128x128 for 16x16 input
        best=np.max(sim, axis=1) # this is the best match to any line for each output neuron, shape number_of_outputs. if 1, perfect line detector, if 0.7, somewhat like a line, if 0.5 or below, not really a line.
        winners=np.argmax(sim, axis=1)# match to which pattern? 
        # note: len(np.unique(winners))/len(winners), this is the coverage, if this value is 1, all neurons match different line types, not the same line. If all neurons detect the same line, coverage is 1/16 = 0.0625 for 8x8
        score=np.mean(best)
        coverage=len(np.unique(winners))/len(winners)
        return score,coverage,best,winners
    def score_sem(result):
        return np.std(result["best"],ddof=1)/np.sqrt(len(result["best"]))
    def activity_sem(result, n=200):
        a=np.asarray(result["activity"][-n:])
        return np.std(a,ddof=1)/np.sqrt(len(a))
    def learning_rate_note(result):
        recent=result["activity"][-200:]
        if result["score"]>0.8 and result["coverage"]>0.7 and np.std(recent)<0.1:
            return "good line detectors"
        if result["score"]>0.7 and result["coverage"]>0.5:
            return "partly learns lines"
        if np.mean(recent)<0.03:
            return "too silent"
        if np.mean(recent)>0.4:
            return "too active"
        if np.std(recent)>0.2:
            return "unstable activity"
        return "weak line detectors"
    def run_model(size=(8,8), p_line=1/8, p=1/8, alpha=0.1, beta=0.02, gamma=0.02, num_batches=1500, settling_steps=300, batch_size=1, seed=0, gammatuned=False):
        np.random.seed(seed)
        number_of_inputs=size[0]*size[1]
        number_of_outputs=size[0]+size[1]
        #todo, more parameters
        dt=0.01
        lambda_=10
        thickness=1
        background=0
        pretrain_patterns=100
        allimg=funs.make_batches_line_pattern(num_batches=num_batches*batch_size, size=size, p_line=p_line, thickness=thickness, background=background)
        qij,wij,ti,activity=funs.train_foldiak(allimg.reshape(allimg.shape[0],-1), number_of_outputs, p, alpha, beta, gamma, lambda_, dt, settling_steps, num_batches, pretrain_patterns, batch_size,gammatuned=gammatuned)
        score,coverage,best,winners=funs.line_score(qij, size)
        return {"qij":qij, "size":size, "p_line":p_line, "p":p, "alpha":alpha, "beta":beta, "gamma":gamma, "activity":np.array(activity), "score":score, "coverage":coverage, "best":best, "winners":winners}
    def run_configs(configs, show_note=False):
        results=[]
        for cfg in configs:
            result=funs.run_model(size=cfg["size"], p_line=cfg["p_line"], p=cfg["p"], alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], num_batches=cfg["num_batches"], settling_steps=cfg["settling_steps"], batch_size=cfg.get("batch_size",1), seed=cfg.get("seed",0), gammatuned=cfg.get("gammatuned",False))
            result["name"]=cfg["name"]
            results.append(result)
            note=f", note={funs.learning_rate_note(result)}" if show_note else ""
            print(f"{cfg['name']}: alpha={cfg['alpha']}, beta={cfg['beta']}, gamma={cfg['gamma']}, score={result['score']:.3f}, coverage={result['coverage']:.3f}, mean_activity={np.mean(result['activity'][-200:]):.3f}{note}")
        return results
    def save_fig(fig, save_dir, script_name, run_tag, name):
        save_path=os.path.join(save_dir, f"{script_name}_{run_tag}_{name}.png")
        if not os.path.exists(save_path):
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
    def sample_natural_patches(vm, num_patches, xdim=16, ydim=16, dtype=None):
        edgeBuff=5
        spRange_x=vm.shape[0]-xdim-edgeBuff*2
        spRange_y=vm.shape[1]-ydim-edgeBuff*2
        spRange_t=vm.shape[2]
        patches=np.zeros((num_patches,xdim*ydim), dtype=dtype)
        for i in range(num_patches):
            xIdx=np.floor(np.random.rand()*spRange_x+edgeBuff).astype(int)
            yIdx=np.floor(np.random.rand()*spRange_y+edgeBuff).astype(int)
            sIdx=np.floor(np.random.rand()*spRange_t).astype(int)
            patch=vm[xIdx:xIdx+xdim,yIdx:yIdx+ydim,sIdx].reshape(-1)
            if dtype is not None:
                patch=patch.astype(dtype)
            patch=patch-np.mean(patch) # zscore
            patch=patch/(np.std(patch)+1e-6) 
            patch=np.clip(patch,-3,3)/3 # large amplitude removal,contrast processing
            patches[i]=patch
        return patches
    def zca_whiten(train_x, test_x, eps=0.1):
        mean_x=np.mean(train_x, axis=0, keepdims=True)
        train0=train_x-mean_x
        test0=test_x-mean_x
        cov=(train0.T@train0)/train0.shape[0]
        eigval,eigvec=np.linalg.eigh(cov)
        W=eigvec@np.diag(1/np.sqrt(eigval+eps))@eigvec.T
        train_w=train0@W
        test_w=test0@W
        train_w=train_w/(np.std(train_w)+1e-6)
        test_w=test_w/(np.std(test_w)+1e-6)
        train_w=np.clip(train_w,-3,3)/3
        test_w=np.clip(test_w,-3,3)/3
        return train_w,test_w,W,mean_x
    def offdiag_mean_abs(mat):
        mask=~np.eye(mat.shape[0], dtype=bool)
        return np.nanmean(np.abs(mat[mask]))
    def output_corr(all_y):
        active=np.std(all_y, axis=0)>1e-6
        if np.sum(active)<2:
            return np.nan
        corr=np.corrcoef(all_y[:,active].T)
        return funs.offdiag_mean_abs(corr)
    def filter_similarity(qij):
        q=qij-np.mean(qij, axis=1, keepdims=True)
        q=q/(np.linalg.norm(q, axis=1, keepdims=True)+1e-12)
        sim=q@q.T
        return funs.offdiag_mean_abs(sim)
    def calibrate_initial_threshold(vm, qij, ti, p, batch_size, xdim, ydim, calibrate_threshold_batches=40):
        all_drive = []
        for _ in range(calibrate_threshold_batches):
            x_raw = funs.sample_natural_patches(vm, batch_size, xdim, ydim, dtype=np.float32)
            xj = funs.onoff_encode(x_raw, dtype=np .float32).T # becacuse in foldriak network, we only have + weights, this processsing as -. 
            all_drive.append(qij @ xj)  
        all_drive = np.concatenate(all_drive, axis=1)
        ti[:, 0] = np.quantile(all_drive, 1.0 - p, axis=1).astype(np.float32)
        return ti
    def plot_falconbridge_components(qij, patch_dim, xdim, ydim, save_dir, script_name, run_tag, name, nrow=12, ncol=12):
        # qij 2048 x 512
        comp = funs.component_from_onoff(qij, patch_dim)
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 10), dpi=220, facecolor="white")
        axes = np.array(axes).reshape(-1)
        for i, ax in enumerate(axes):
            if i < comp.shape[0]:
                ax.imshow(comp[i].reshape(xdim, ydim), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
            ax.axis("off")
        plt.tight_layout(pad=0.15)
        save_path = os.path.join(save_dir, f"{script_name}_{run_tag}_{name}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path
    def plot_falconbridge_activity(activity, p, save_dir, script_name, run_tag, name):
        fig, ax = plt.subplots(figsize=(8, 3), dpi=220, facecolor="white")
        a = np.array(activity, dtype=np.float32)
        if len(a) > 50:
            smooth = np.convolve(a, np.ones(50) / 50, mode="valid")
            ax.plot(smooth, lw=1.5)
        else:
            ax.plot(a, lw=1.5)
        ax.axhline(p, color="0.35", lw=1, ls="--")
        ax.set_title("mean output activity")
        ax.set_xlabel("update batch")
        ax.set_ylabel("mean y")
        ax.grid(axis="y", color="0.88", lw=0.8)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{script_name}_{run_tag}_{name}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path
    def evaluate_falconbridge_network(vm, qij, wij, ti, patch_dim, xdim, ydim, batch_size, lambda_, dt, settling_steps, clip_sigmoid_input=None, num_test=1000):
        all_y = []
        for _ in range(int(np.ceil(num_test / batch_size))):
            x_raw = funs.sample_natural_patches(vm, batch_size, xdim, ydim, dtype=np.float32)
            xj = funs.onoff_encode(x_raw, dtype=np.float32).T
            y = funs.settling_y(qij, wij, ti, xj, lambda_=lambda_, dt=dt, settling_steps=settling_steps, clip_sigmoid_input=clip_sigmoid_input, dtype=np.float32)
            all_y.append(y.T) # can be (100, 2048)
        all_y = np.concatenate(all_y, axis=0)[:num_test] # (images, ouputs)
        comp = funs.component_from_onoff(qij, patch_dim)
        sparsity = float(np.mean(all_y))
        corr = funs.output_corr(all_y)
        similarity = funs.filter_similarity(comp)
        dead_units = int(np.sum(np.mean(all_y, axis=0) < 1e-4))
        return sparsity, corr, similarity, dead_units, all_y
    def load_falconbridge_feature_runs(source_dir, bruno_paths, patch_dim):
        import glob
        runs=[]
        for path in sorted(glob.glob(os.path.join(source_dir,"N*_size16_onoff_*","result_final.npz"))):
            data=np.load(path)
            run_dir=os.path.dirname(path)
            name=os.path.basename(run_dir).split("_onoff_")[0]
            qij=data["qij"]
            rf=funs.component_from_onoff(qij,patch_dim)
            mean_cos,nn90=funs.cosine_stats(rf)
            runs.append({"name":name,"path":path,"rf":rf,"rf_concentration":funs.top_energy_fraction(rf),"mean_cosine":mean_cos,"nn90_cosine":nn90,"sparsity":float(data["sparsity"]),"output_corr":float(data["corr"]),"dead_units":int(data["dead_units"])})
        for bruno_path in bruno_paths:
            if os.path.exists(bruno_path):
                basis=np.load(bruno_path)
                rf=basis.T if basis.shape[0]==patch_dim else basis
                rf=rf-np.mean(rf,axis=1,keepdims=True)
                rf=rf/(np.max(np.abs(rf),axis=1,keepdims=True)+1e-12)
                mean_cos,nn90=funs.cosine_stats(rf)
                runs.append({"name":f"Bruno_N{rf.shape[0]}","path":bruno_path,"rf":rf,"rf_concentration":funs.top_energy_fraction(rf),"mean_cosine":mean_cos,"nn90_cosine":nn90,"sparsity":np.nan,"output_corr":np.nan,"dead_units":np.nan})
        return runs
    def plot_summary(results, title, save_path=None):
        names=[r["name"] for r in results]
        scores=[r["score"] for r in results]
        coverages=[r["coverage"] for r in results]
        x=np.arange(len(results))
        fig,ax=plt.subplots(figsize=(max(9,1.5*len(results)),3.2), dpi=220, facecolor="white")
        ax.bar(x-0.18, scores, width=0.36, yerr=[funs.score_sem(r) for r in results], capsize=2.4, color="#4f6fd5", edgecolor="white", linewidth=.8, alpha=.88, label="line score")
        ax.bar(x+0.18, coverages, width=0.36, color="0.55", edgecolor="white", linewidth=.8, alpha=.82, label="coverage")
        ax.axhline(0.8, color="#d62728", lw=1.1, ls="--", alpha=.75)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0,1.05)
        ax.set_ylabel("score")
        ax.set_title(title)
        ax.legend(frameon=False, ncols=2)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="0.88", lw=.8)
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not os.path.exists(save_path):
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    def plot_activity(results, title, save_path=None):
        colors=["0.12","#4f6fd5","#d62728","0.45","#145da0","#f58518","#6f8fe8","0.65"]
        fig,ax=plt.subplots(figsize=(max(9,1.5*len(results)),3.2), dpi=220, facecolor="white")
        for i,r in enumerate(results):
            ax.plot(np.convolve(r["activity"], np.ones(80)/80, mode="valid"), color=colors[i%len(colors)], lw=1.8, label=r["name"])
        ax.axhline(results[0]["p"], color="0.55", lw=.9, ls="--", label="target p")
        ax.set_xlabel("training step")
        ax.set_ylabel("mean activity")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7, ncols=2)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(color="0.9", lw=.8)
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not os.path.exists(save_path):
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    def plot_qij(result, title=None, save_path=None, show=True):
        qij=result["qij"]
        size=result["size"]
        number_of_outputs=qij.shape[0]
        ncols=8
        nrows=int(np.ceil(number_of_outputs/ncols))
        fig,axes=plt.subplots(nrows,ncols,figsize=(1.35*ncols,1.35*nrows), dpi=220, facecolor="white")
        axes=np.array(axes).reshape(-1)
        vmin=np.percentile(qij, 2)
        vmax=np.percentile(qij, 98)
        for i,ax in enumerate(axes):
            ax.axis("off")
            if i<number_of_outputs:
                ax.imshow(qij[i,:].reshape(size), cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
                ax.add_patch(Rectangle((-0.5,-0.5), size[1], size[0], fill=False, edgecolor="#d62728", linewidth=1.1))
                ax.set_title(f"{i}: {result['best'][i]:.2f}", fontsize=7, pad=2)
        fig.suptitle(title if title else f"{result['name']} | score={result['score']:.3f}, coverage={result['coverage']:.3f}", fontsize=13, y=0.99)
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not os.path.exists(save_path):
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
