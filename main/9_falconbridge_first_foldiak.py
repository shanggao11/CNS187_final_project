#%%
import numpy as np
import matplotlib.pyplot as plt
from funs import funs
import os
import time
np.random.seed(0)
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
script_name = "9_falconbridge_first_foldiak"
save_dir = os.path.join("Results", script_name)
os.makedirs(save_dir, exist_ok=True)
dataload_path = "./data/bruno_sparse/IMAGES_Vanhateren.npy"
xdim = 16
ydim = 16
patch_dim = xdim * ydim
number_of_outputs = 2048

p = 0.005                  # paper alpha, desired mean activity
lambda_ = 200.0            # paper beta, sigmoid steepness
alpha = 2.4                # eta2, lateral anti-Hebbian learning rate
gamma = 1.5                # eta3, threshold/sensitivity learning rate
eta1_schedule = [(3.0, 1000000), (1.5, 400000), (0.725, 1000000)]  # eta1 by image presentations, eta here is the one in hebbina 

# paper says batch averages can be used; this converts presentations to update batches
batch_size = 100
settling_steps = 120
dt = 0.08
edgeBuff = 5
print_every = 100
save_every = 1000
calibrate_threshold_batches = 40

# numerical options
clip_sigmoid_input = 50.0
use_fast_debug = False
if use_fast_debug:
    eta1_schedule = [(3.0, 100000)]
    settling_steps = 80
    print_every = 20
    save_every = 100

run_tag = f"N{number_of_outputs}_size{xdim}_onoff_p{p:g}_eta2{alpha:g}_eta3{gamma:g}_lam{lambda_:g}_bs{batch_size}"
one_save_dir = os.path.join(save_dir, run_tag)
os.makedirs(one_save_dir, exist_ok=True)

#%%
vm = np.load(dataload_path)
print("vm", vm.shape)
number_of_inputs = patch_dim * 2 # becuaes foldriak does not allow negative pixels (not like sparse coding), here we use on-off as approximation. 
qij, wij, ti = funs.initialization(number_of_outputs, number_of_inputs, threshold_init="zero", dtype=np.float32, return_order="qwt")
ti = funs.calibrate_initial_threshold(vm, qij, ti, p, batch_size, xdim, ydim, calibrate_threshold_batches)
print("qij", qij.shape, "wij", wij.shape, "ti", ti.shape)

activity = []
update_count = 0
presentation_count = 0
start_time = time.time()

init_path = funs.plot_falconbridge_components(qij, patch_dim, xdim, ydim, one_save_dir, script_name, run_tag, "components_init") # This is plotting the raw weights (with negative pixels)
print("saved", init_path)

#%%
for phase_id, (beta_now, phase_presentations) in enumerate(eta1_schedule):
    phase_updates = int(np.ceil(phase_presentations / batch_size))
    print(f"start phase {phase_id}: eta1/beta={beta_now}, presentations={phase_presentations}, updates={phase_updates}")
    for sstep in range(phase_updates):
        x_raw = funs.sample_natural_patches(vm, batch_size, xdim, ydim, dtype=np.float32)
        xj = funs.onoff_encode(x_raw, dtype=np.float32).T
        yj_star = funs.settling_y(qij, wij, ti, xj, lambda_=lambda_, dt=dt, settling_steps=settling_steps, clip_sigmoid_input=clip_sigmoid_input, dtype=np.float32)
        y = yj_star
        qij, wij, ti = funs.update_weights(qij, wij, ti, xj, y, alpha, beta_now, gamma, p, q_decay="second_moment", dtype=np.float32) # secondmmoment here means oja's rule, basically y**2 x qij
        if not funs.is_finite(qij, wij, ti, y):
            raise ValueError(f"NaN/inf at phase={phase_id}, step={sstep}; reduce dt, eta1, eta2, or eta3")
        mean_y = float(np.mean(y))
        activity.append(mean_y)
        update_count += 1
        presentation_count += batch_size

        if update_count % print_every == 0:
            y_unit = np.mean(y, axis=1)
            elapsed = (time.time() - start_time) / 60
            print(f"update={update_count}, presentations={presentation_count}, phase={phase_id}, eta1={beta_now:g}, mean_y={mean_y:.5f}, unit_y_min={y_unit.min():.5f}, unit_y_max={y_unit.max():.5f}, ti_mean={float(np.mean(ti)):.4f}, u_mean={float(np.mean(wij)):.4f}, elapsed_min={elapsed:.1f}")

        if update_count % save_every == 0:
            sparsity, corr, similarity, dead_units, y_test = funs.evaluate_falconbridge_network(vm, qij, wij, ti, patch_dim, xdim, ydim, batch_size, lambda_, dt, settling_steps, clip_sigmoid_input, num_test=1000)
            print(f"eval: sparsity={sparsity:.5f}, output_corr={corr:.5f}, filter_similarity={similarity:.5f}, dead_units={dead_units}")
            funs.plot_falconbridge_components(qij, patch_dim, xdim, ydim, one_save_dir, script_name, run_tag, f"components_update{update_count}")
            funs.plot_falconbridge_activity(activity, p, one_save_dir, script_name, run_tag, f"activity_update{update_count}")
            np.savez_compressed(os.path.join(one_save_dir, f"checkpoint_update{update_count}.npz"), qij=qij, wij=wij, ti=ti, activity=np.array(activity), y_test=y_test, sparsity=sparsity, corr=corr, similarity=similarity, dead_units=dead_units, update_count=update_count, presentation_count=presentation_count)

#%%
sparsity, corr, similarity, dead_units, y_test = funs.evaluate_falconbridge_network(vm, qij, wij, ti, patch_dim, xdim, ydim, batch_size, lambda_, dt, settling_steps, clip_sigmoid_input, num_test=5000)
final_grid_path = funs.plot_falconbridge_components(qij, patch_dim, xdim, ydim, one_save_dir, script_name, run_tag, "components_final")
activity_path = funs.plot_falconbridge_activity(activity, p, one_save_dir, script_name, run_tag, "activity_final")
np.savez_compressed(os.path.join(one_save_dir, "result_final.npz"), qij=qij, wij=wij, ti=ti, activity=np.array(activity), y_test=y_test, sparsity=sparsity, corr=corr, similarity=similarity, dead_units=dead_units, update_count=update_count, presentation_count=presentation_count)

print("Finish...")
print("final_grid_path", final_grid_path)
print("activity_path", activity_path)
print(f"final metrics: sparsity={sparsity:.5f}, output_corr={corr:.5f}, filter_similarity={similarity:.5f}, dead_units={dead_units}")
print("result dir", one_save_dir)
