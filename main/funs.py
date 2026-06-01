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

    def update_weights(qij, wij, ti, xj, y, alpha, beta, gamma, p):
        batch_size = xj.shape[1]
        yiyj= (y @ y.T) / batch_size
        yi_mean = np.mean(y, axis=1, keepdims=True)
        yixj = (y @ xj.T) / batch_size
        delta_w = -alpha * (yiyj - p**2)
        delta_ti = gamma * (yi_mean - p)
        # delta_qij = beta * y * (xj.T - qij)
        delta_qij = beta * (yixj - yi_mean * qij)
        wij = wij + delta_w
        np.fill_diagonal(wij, 0)
        wij = np.minimum(wij, 0)
        ti = ti + delta_ti
        qij = qij + delta_qij
        return qij, wij, ti
    def settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps):
        feedforward_i = qij @ xj - ti
        for _ in range(settling_steps):
            dystar_i= (funs.sigmoid(feedforward_i + wij@yj_star, lambda_) - yj_star) * dt
            yj_star += dystar_i
        return yj_star
    def initialization(number_of_outputs, number_of_inputs):
        t_init_i=(np.random.rand(number_of_outputs) * 0.01 - 0.005).reshape(number_of_outputs,1) # small random values for initial thresholds
        w_init=np.zeros((number_of_outputs, number_of_outputs))
        q_init = np.random.rand(number_of_outputs, number_of_inputs)
        q_init = q_init /np.linalg.norm(q_init, axis=1, keepdims=True) # every row is output neuron, every column is input to that output neuorn. 
        return t_init_i,w_init,q_init

    def sigmoid(x,lambda_=10):
        y=1/(1+np.exp(-lambda_*x))
        return y
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
        best=np.max(sim, axis=1) # this is the best match to any line for each output neuron, shape number_of_outputs. if 1, perfect line detector, if 0.7, somewhat like a line, if 0.5 or below, not really a line.
        winners=np.argmax(sim, axis=1)
        # note: len(np.unique(winners))/len(winners), this is the coverage, if this value is 1, all neurons match different line types, not the same line. If all neurons detect the same line, coverage is 1/16 = 0.0625 for 8x8
        score=np.mean(best)
        coverage=len(np.unique(winners))/len(winners)
        return score,coverage,best,winners
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
    def run_model(size=(8,8), p_line=1/8, p=1/8, alpha=0.1, beta=0.02, gamma=0.02, num_batches=1500, settling_steps=300, seed=0):
        np.random.seed(seed)
        number_of_inputs=size[0]*size[1]
        number_of_outputs=size[0]+size[1]
        #todo, more parameters
        dt=0.01
        lambda_=10
        thickness=1
        background=0
        batch_size=1
        pretrain_patterns=100
        pretrain_updates=pretrain_patterns//batch_size
        t_init_i,w_init,q_init=funs.initialization(number_of_outputs, number_of_inputs)
        allimg=funs.make_batches_line_pattern(num_batches=num_batches*batch_size, size=size, p_line=p_line, thickness=thickness, background=background)
        wij=w_init.copy()
        ti=t_init_i.copy()
        qij=q_init.copy() #  number_of_outputs x number_of_inputs
        activity=[]
        for sstep in range(num_batches):
            xj=allimg[sstep*batch_size:(sstep+1)*batch_size].reshape(batch_size,-1).T
            yj_star=np.zeros((number_of_outputs,batch_size))
            yj_star=funs.settling_y(qij, wij, ti, xj, yj_star, lambda_, dt, settling_steps)
            y=funs.binarize(yj_star)
            alpha_now,beta_now,gamma_now=(0.0,0.0,0.1) if sstep<pretrain_updates else (alpha,beta,gamma)
            qij,wij,ti=funs.update_weights(qij=qij, wij=wij, ti=ti, xj=xj, y=y, alpha=alpha_now, beta=beta_now, gamma=gamma_now, p=p)
            activity.append(np.mean(y))
        score,coverage,best,winners=funs.line_score(qij, size)
        return {"qij":qij, "size":size, "p_line":p_line, "p":p, "alpha":alpha, "beta":beta, "gamma":gamma, "activity":np.array(activity), "score":score, "coverage":coverage, "best":best, "winners":winners}
    def run_configs(configs, show_note=False):
        results=[]
        for cfg in configs:
            result=funs.run_model(size=cfg["size"], p_line=cfg["p_line"], p=cfg["p"], alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], num_batches=cfg["num_batches"], settling_steps=cfg["settling_steps"], seed=cfg.get("seed",0))
            result["name"]=cfg["name"]
            results.append(result)
            note=f", note={funs.learning_rate_note(result)}" if show_note else ""
            print(f"{cfg['name']}: alpha={cfg['alpha']}, beta={cfg['beta']}, gamma={cfg['gamma']}, score={result['score']:.3f}, coverage={result['coverage']:.3f}, mean_activity={np.mean(result['activity'][-200:]):.3f}{note}")
        return results
    def save_fig(fig, save_dir, script_name, run_tag, name):
        save_path=os.path.join(save_dir, f"{script_name}_{run_tag}_{name}.png")
        if not os.path.exists(save_path):
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
    def sample_natural_patches(vm, num_patches, xdim=16, ydim=16):
        edgeBuff=5
        spRange_x=vm.shape[0]-xdim-edgeBuff*2
        spRange_y=vm.shape[1]-ydim-edgeBuff*2
        spRange_t=vm.shape[2]
        patches=np.zeros((num_patches,xdim*ydim))
        for i in range(num_patches):
            xIdx=np.floor(np.random.rand()*spRange_x+edgeBuff).astype(int)
            yIdx=np.floor(np.random.rand()*spRange_y+edgeBuff).astype(int)
            sIdx=np.floor(np.random.rand()*spRange_t).astype(int)
            patch=vm[xIdx:xIdx+xdim,yIdx:yIdx+ydim,sIdx].reshape(-1)
            patch=patch-np.mean(patch) # zscore
            patch=patch/(np.std(patch)+1e-6)
            patch=np.clip(patch,-3,3)/3
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
    def plot_summary(results, title, save_path=None):
        names=[r["name"] for r in results]
        scores=[r["score"] for r in results]
        coverages=[r["coverage"] for r in results]
        x=np.arange(len(results))
        fig,ax=plt.subplots(figsize=(max(9,1.5*len(results)),3.2), dpi=220, facecolor="white")
        ax.bar(x-0.18, scores, width=0.36, color="#4f6fd5", edgecolor="white", linewidth=.8, alpha=.88, label="line score")
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
    def plot_qij(result, title=None, save_path=None):
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
        plt.show()
