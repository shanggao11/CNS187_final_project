#%%
import numpy as np
from funs import funs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

script_name="1_reproduce_figure2_3"
save_dir=os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
size=(8,8)
p_line=1/8 #1/8
number_of_inputs=size[0]*size[1]
number_of_outputs= size[0]+size[1]
# number_of_outputs=size[0]*size[1]*2
numberoflayers=1
alpha=0.1
beta=0.02
gamma=0.02 #0.02 default
dt=0.01
# p=p_line
p=1/8
lambda_=10
num_batches = 1500
settling_steps=300
thickness=1
background=0
batch_size = 1
pretrain_patterns = 100
num_images = num_batches * batch_size
# checkpoints = [0, 1, 5, 10, num_batches-1]
checkpoints = [0, 400, 800, 1200, num_batches-1]
checkpoints = [c for c in checkpoints if c < num_batches]
#%
allimg= funs.make_batches_line_pattern(num_batches=num_images, size=size, p_line=p_line, thickness=thickness, background=background)

qij,wij,ti,activity,qij_checkpoints=funs.train_foldiak(allimg.reshape(allimg.shape[0],-1), number_of_outputs, p, alpha, beta, gamma, lambda_, dt, settling_steps, num_batches, pretrain_patterns, batch_size, checkpoints,gammatuned=False)

#%%

fig, axes = plt.subplots(len(checkpoints), number_of_outputs, figsize=(20, 0.9*len(checkpoints)))
for r, step in enumerate(checkpoints):
    for i in range(number_of_outputs):
        ax = axes[r, i]
        ax.imshow(qij_checkpoints[step][i, :].reshape(size[0], size[1]), cmap="gray")
        ax.add_patch(Rectangle((-0.5, -0.5), size[0], size[1], fill=False, edgecolor="red", linewidth=1.2))
        ax.axis("off")
        if r == 0:
            ax.set_title(f"{i}", fontsize=7, pad=3)
    y_pos = 1 - (r + 0.5) / len(checkpoints)
    fig.text(0.035, y_pos, f"{step}", ha="right", va="center", fontsize=20)
plt.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.04, wspace=0.25, hspace=0.25)
print("Evolution of feedforward weights (qij) at different training steps")
save_path=os.path.join(save_dir, f"{script_name}_qij_evolution.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
# %%
n_show = 16
fig, axes = plt.subplots(1, n_show, figsize=(12, 1))
for i, ax in enumerate(axes):
    ax.imshow(allimg[i], cmap="gray", interpolation="nearest")
    ax.set_title(f"{i}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
save_path=os.path.join(save_dir, f"{script_name}_example_inputs.png")
if not os.path.exists(save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
print("Example input patterns")
