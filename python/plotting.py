import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import imageio
import shutil
import random
import numpy as np
import torch
import os
from utils import get_device
from state_dataset import StateDataset
from pomdp import State, generate_true_state
from policies import simulate

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # Use Latin Modern instead
    "font.size": 12,
    "mathtext.fontset": "stix", # "cm"
})

plt.style.use('dark_background')
colors = ["#44342a", "#e1e697"]  # Dark brown to tan (#3E1F00 to #D2B48C)
cmap = LinearSegmentedColormap.from_list("brown_to_tan", colors, N=256)


def disable_axes():
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])


def square_axes(ax=None):
    if ax == None:
        ax = plt.gca()
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.set_aspect(x_range / y_range, adjustable='box')


def plot_state(s, cmap=cmap):
    fig = plt.figure()
    fig.patch.set_alpha(0)
    plt.imshow(s.reshape(80, 80), cmap=cmap, vmin=0, vmax=1)
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.tight_layout()


def plot_prior(x_train, n=100, seed=4, cmap=cmap):
    random.seed(seed)
    fig = plt.figure(figsize=(6,6))
    fig.patch.set_alpha(0)
    r = c = int(np.sqrt(n))
    for i in range(n):
        plt.subplot(r,c,i+1)
        idx = random.randint(0, len(x_train))
        plt.imshow(x_train[idx].squeeze(0), cmap=cmap)
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
    plt.show()


def plot_belief(b, isnp=False, reduce='mean', cmap=cmap):
    fig = plt.figure()
    fig.patch.set_alpha(0)
    if isnp:
        if reduce == 'mean':
            b_reduce = np.mean(b, axis=0)
        elif reduce == 'std':
            b_reduce = np.std(b, axis=0)
        elif reduce == 'var':
            b_reduce = np.var(b, axis=0)
    else:
        if reduce == 'mean':
            b_reduce = torch.mean(b, dim=0).detach().cpu()
        elif reduce == 'std':
            b_reduce = torch.std(b, dim=0).detach().cpu()
        elif reduce == 'var':
            b_reduce = torch.var(b, dim=0).detach().cpu()
    plt.imshow(b_reduce.reshape(80,80), cmap=cmap)
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.tight_layout()


def plot_obs(o, cmap=cmap, vmin=0):
    fig = plt.figure()
    plt.imshow(o.detach().cpu().reshape(200,200), origin='lower', vmin=vmin, vmax=1, cmap=cmap)
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    return fig


def plot_obs_and_state(x_test, precomputed_muon_test, test_idx, cmap=cmap):
    plt.subplots(1,2)

    plt.subplot(1,2,1)
    M = precomputed_muon_test[test_idx]
    plt.imshow(np.flipud(M.reshape(10,10,20,20).transpose(0,2,1,3).reshape(200,200)), cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1,2,2)
    s_train = torch.Tensor(x_test[test_idx])
    plt.imshow(s_train.reshape(80, 80), cmap=cmap, vmin=0, vmax=1)
    plt.subplots_adjust(right=1.5)


def plot_obs_surrogate(obs_model, model, x_test, x_test3d, muon_data, muon_test_data, s_idx=2, output_suffix="", use_surrogate=True, print_stats=False, thresholded=False, m=100, mass_center=np.inf, save=False):

    obs_test_dataset = StateDataset(data=x_test, data3d=x_test3d, muon_data=muon_data, muon_test_data=muon_test_data, num_samples=900, num_observations=100, force_num_obs=True, is_test=True)

    true_state, true_obs = obs_test_dataset.__getitem__(s_idx, use_index=True)

    device = get_device(obs_model)
    est_obs = obs_model(true_state.unsqueeze(0).to(device))

    plt.subplots(1,6,figsize=(19,4))

    initial_obs = torch.full((1,1,200,200), -1.0).to(next(model.parameters()).device)
    S = model.sample(initial_obs, m=1000)
    plt.subplot(1,6,1)
    plt.imshow(torch.mean(S, dim=0).detach().cpu().reshape(80,80), cmap='viridis', vmin=0, vmax=1)
    plt.gca().set_title('$\mathcal{I}$-VAE initial belief')
    disable_axes()

    plt.subplot(1,6,2)
    plt.imshow(true_state.detach().cpu().reshape(80,80), cmap=cmap)
    # plt.gca().axis('off')
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().set_title("true state")

    try:
        if use_surrogate:
            est_belief_states = model.sample(est_obs, thresholded=thresholded, m=m)
        else:
            est_belief_states = model.sample(true_obs.unsqueeze(0).to(device), thresholded=thresholded, m=m)
        est_belief_state = torch.mean(est_belief_states, dim=0)
        est_belief_state_title = "belief mean"
    except:
        est_belief_states = torch.zeros((m, 1, 80,80))
        est_belief_state = torch.zeros((80,80))
        est_belief_state_title = "(model not defined)"
    est_belief_state_var = torch.var(est_belief_states, dim=0)

    plt.subplot(1,6,3)
    plt.imshow(est_belief_state.detach().cpu().reshape(80,80), cmap='hot')
    # plt.gca().axis('off')
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().set_title(est_belief_state_title)

    plt.subplot(1,6,4)
    plt.imshow(est_belief_state_var.detach().cpu().reshape(80,80), cmap='hot')
    # plt.gca().axis('off')
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().set_title('belief variance')

    plt.subplot(1,6,5)
    plt.imshow(true_obs.detach().cpu().reshape(200,200), origin='lower', vmin=0, vmax=1, cmap='magma')
    # plt.imshow(true_obs, origin='lower', vmin=0, vmax=1)
    # plt.gca().axis('off')
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().set_title("muon observation")

    plt.subplot(1,6,6)
    plt.imshow(est_obs.detach().cpu().reshape(200,200), origin='lower', vmin=0, vmax=1, cmap='magma')
    # plt.gca().axis('off')
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().set_title("surrogate muon observation")

    if save:
        plt.savefig(f"muon-surrogate-{s_idx}{output_suffix}.png")

    if print_stats:
        print(f"Truth: {torch.sum(true_state.reshape(80,80)) - mass_center:.4g}")
        print(f"Belief: {torch.mean(torch.sum(est_belief_states, dim=[2,3]) - mass_center):.4g} ± {torch.std(torch.sum(est_belief_states, dim=[2,3]) - mass_center):.4g}")
        # print(f"Belief: {torch.sum(est_belief_state.reshape(80,80)) - mass_center:.4g} ± {torch.std(torch.sum(est_belief_states, dim=[2,3]) - mass_center):.4g}")
    

def plot_simulation_error(infos, labels, colors):
    plt.figure()
    for i in range(len(infos)):
        plt.plot(infos[i]['errors'], label=labels[i], color=colors[i])
    plt.legend()


def plot_belief_over_time(beliefs, test_state, cmap=cmap):
    n_beliefs = np.min([5, len(beliefs)]) + 2 # +2 for true state and final belief
    plt.subplots(1, n_beliefs)
    plt.subplot(1, n_beliefs, 1)
    plt.imshow(test_state.squeeze(0), cmap=cmap)
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])

    for i in range(n_beliefs-2):
        plt.subplot(1, n_beliefs, i+2)
        plt.imshow(torch.mean(beliefs[i], dim=0).squeeze(0).detach().cpu(), cmap=cmap)
        plt.gca().set_title(f't = {i}')
        plt.gca().axis('off')

    plt.subplot(1, n_beliefs, n_beliefs)
    plt.imshow(torch.mean(beliefs[-1], dim=0).squeeze(0).detach().cpu(), cmap=cmap)
    plt.gca().set_title(f't = {len(beliefs)}')
    plt.gca().axis('off')
    plt.show()


def plot_qualitative(
        pomdp,
        info,
        state_obj,
        m=900,
        time_steps=[0,1,2,3,4],
        show_obs=True,
        zoomed=False,
        zoom_offset=100,
        include_std=False,
        include_conf_bounds=False,
        lambd=1,
        cmap=cmap,
        save=False,
        seed=1):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = get_device(pomdp.belief_updater)
    mass_center = pomdp.mass_center

    # plt.style.use('default')
    plt.style.use('dark_background')

    num_rows = 3 if include_std else 2
    plt.subplots(num_rows, 7, figsize=(16,8 if include_std else 6), facecolor="black", dpi=1000)

    time_steps = [*time_steps, len(info['obs'])-1]

    for i in range(6):
        t = time_steps[i]
        sample_obs = info['obs'][t].to(device)
        generated_states = pomdp.belief_updater.sample(sample_obs, m=m)

        plt.subplot(num_rows,7,i+1)
        plt.imshow(torch.mean(generated_states, dim=0).detach().cpu().reshape(80,80), cmap=cmap)
        # plt.gca().axis('off')
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.gca().set_title(rf"$t = {t}$")
        if i == 0:
            plt.gca().set_ylabel("belief mean")

        if show_obs:
            locs = (np.linspace(200, 3800, 10) / 50).astype(int)
            x_loc = [80 - locs[xyv[1]] for xyv in info['b'][t].history] # NOTE: 80 - location for "origin=lower"
            y_loc = [locs[xyv[0]] for xyv in info['b'][t].history] # NOTE: 1-0 flip
            v_val = [1 for xyv in info['b'][t].history]
            for j in range(t):
                hit = v_val[j] == 1
                c = "crimson"
                marker = "x" if hit else "o"
                plt.scatter([y_loc[j]], [x_loc[j]], marker=marker, c=c, s=8, alpha=0.5 if t == 100 else 1.0)


        if include_std:
            plt.subplot(num_rows,7,i+1+7)
            plt.imshow(torch.std(generated_states, dim=0).detach().cpu().reshape(80,80), cmap=cmap)
            # plt.gca().axis('off')
            plt.gca().get_xaxis().set_ticks([])
            plt.gca().get_yaxis().set_ticks([])
            if i == 0:
                plt.gca().set_ylabel("belief std.")


        plt.subplot(num_rows,7,i+1+ (14 if include_std else 7))
        massive_ore = torch.sum(generated_states, dim=[1,2,3]).detach().cpu() - mass_center
        plt.hist(massive_ore, color=colors[-1], edgecolor=colors[0])

        plt.axvline(torch.sum(state_obj.state).item() - mass_center, color='#0072FE', label="true volume", linewidth=1.5) # truth

        plt.axvline(torch.mean(massive_ore), color='crimson', linestyle='--', label="est. mean", linewidth=1.5) # estimated mean

        if include_conf_bounds:
            plt.axvline(torch.mean(massive_ore) - lambd*torch.std(massive_ore), color='magenta', linestyle='-.', label="conf. bounds", linewidth=1) # estimated LCB
            plt.axvline(torch.mean(massive_ore) + lambd*torch.std(massive_ore), color='magenta', linestyle='-.', linewidth=1) # estimated UCB
        
        true_mass = torch.sum(state_obj.state).item()
        
        if zoomed:
            if i == 5:
                plt.xlim(true_mass - mass_center - zoom_offset, true_mass - mass_center + zoom_offset)
            else:
                plt.xlim(-600,600)
        else:
            plt.xlim(-600,600)

        sd = 2 if i != 3 else 6
        sorted_data = torch.sort(massive_ore).values # Sort the data to compute the empirical CDF
        greater_than_zero = sorted_data[sorted_data > 0] # Find the rank of the largest value less than 0
        probability_above_zero = len(greater_than_zero) / len(sorted_data) # Empirical CDF at 0
        plt.gca().set_title(rf"$P(v > 0) \,\approx\, {probability_above_zero:.{sd}g}$")

        plt.gca().get_yaxis().set_ticks([])
        plt.gca().set_xlabel("economic volume")
        x_range = plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]
        y_range = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
        plt.gca().set_aspect(x_range / y_range, adjustable='box')

        if i == len(time_steps)-1:
            plt.legend(fancybox=False, frameon=True, fontsize=9)
            print(f"{true_mass - mass_center:0.5f}")
            print(f"{torch.mean(massive_ore).item():0.5f} ± {torch.std(massive_ore).item():0.2f}")

    plt.subplot(num_rows,7,7)
    plt.imshow(state_obj.state.reshape(80,80), cmap=cmap)
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().set_title(f"true state")

    plt.subplot(num_rows,7,14)
    plt.gca().axis('off')

    if include_std:
        plt.subplot(num_rows,7,21)
        plt.gca().axis('off')

    plt.tight_layout()

    if save:
        if zoomed:
            plt.savefig(f"simulation-idx-{state_obj.idx}-zoomed.png")
        else:
            plt.savefig(f"simulation-idx-{state_obj.idx}.png")

    # plt.show()


def create_animation(frames, output_name='video.mp4', fps=3, loops=1, is_gif=False):
    # Create a directory to save frames
    frames_dir = 'frames_temp'
    os.makedirs(frames_dir, exist_ok=True)

    frames = frames * loops

    # Save figures as images
    frame_files = []
    for idx, fig in enumerate(frames):
        filename = os.path.join(frames_dir, f'frame_{idx:04d}.png')
        fig.savefig(filename)
        frame_files.append(filename)
        plt.close(fig)  # Free memory
    
    if os.path.exists(output_name):
        os.remove(output_name)

    if is_gif:
        # Read images and append to a list
        images = []
        for filename in frame_files:
            images.append(imageio.imread(filename))

        # Save as GIF
        imageio.mimsave(output_name, images, fps=fps, loop=0)
    else:
        # Create MP4 using imageio
        with imageio.get_writer(output_name, fps=fps) as writer:
            for filename in frame_files:
                image = imageio.imread(filename)
                writer.append_data(image)

    # Optional: Remove the temporary frames directory
    shutil.rmtree(frames_dir)


def animate_belief_updates(
        pomdp,
        x_test,
        idx=0,
        delta=np.inf,
        t_max=16,
        m_sim=100,
        m=900,
        seed=1,
        show_obs=True):

    s = generate_true_state(x_test, idx=idx)[0]
    state_obj = State(s, idx=idx)
    mass_center = pomdp.mass_center
    device = get_device(pomdp.belief_updater)

    info = simulate(pomdp, state_obj, delta=delta, verbose=False, progress=True, oracle_policy=False, m=m_sim, nsteps=1, seed=idx)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # plt.style.use('default')
    plt.style.use('dark_background')

    frames = []

    if t_max == np.inf:
        t_max = len(info['obs'])

    for i in range(t_max):
        fig, axes = plt.subplots(1, 3, figsize=(6,2), facecolor="black", dpi=200)

        t = i
        sample_obs = info['obs'][t].to(device)
        generated_states = pomdp.belief_updater.sample(sample_obs, m=m)

        plt.subplot(1,3,1)
        plt.imshow(torch.mean(generated_states, dim=0).detach().cpu().reshape(80,80), cmap=cmap)
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.gca().set_title(rf"$t = {t}$")
        plt.gca().set_ylabel("belief mean")

        if show_obs:
            locs = (np.linspace(200, 3800, 10) / 50).astype(int)
            x_loc = [80 - locs[xyv[1]] for xyv in info['b'][t].history] # NOTE: 80 - location for "origin=lower"
            y_loc = [locs[xyv[0]] for xyv in info['b'][t].history] # NOTE: 1-0 flip
            v_val = [1 for xyv in info['b'][t].history]
            for j in range(t):
                hit = v_val[j] == 1
                c = "crimson"
                marker = "x" if hit else "o"
                plt.scatter([y_loc[j]], [x_loc[j]], marker=marker, c=c, s=8, alpha=0.5 if t == 100 else 1.0)

        plt.subplot(1,3,3)
        massive_ore = torch.sum(generated_states, dim=[1,2,3]).detach().cpu() - mass_center # _test # NOTE: Better with `test`
        plt.hist(massive_ore, color=colors[-1], edgecolor=colors[0], linewidth=0.5)

        plt.axvline(torch.sum(s).item() - mass_center, color='#0072FE', label="true volume", linewidth=1) # truth
        plt.axvline(torch.mean(massive_ore), color='crimson', linestyle='--', label="est. mean", linewidth=1) # estimated mean
        plt.xlim(-600,600)

        sd = 3

        sorted_data = torch.sort(massive_ore).values # Sort the data to compute the empirical CDF
        greater_than_zero = sorted_data[sorted_data > 0] # Find the rank of the largest value less than 0
        probability_above_zero = len(greater_than_zero) / len(sorted_data) # Empirical CDF at 0
        plt.gca().set_title(rf"$P(v > 0) \,\approx\, {probability_above_zero:.{sd}g}$", fontsize=12)

        plt.gca().get_yaxis().set_ticks([])
        plt.xticks(fontsize=10)
        # plt.gca().set_xlabel("economic volume")
        x_range = plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]
        y_range = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
        plt.gca().set_aspect(x_range / y_range, adjustable='box')

        plt.subplot(1,3,2)
        plt.imshow(s.reshape(80,80), cmap=cmap)
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.gca().set_title(f"true state")

        frames.append(fig)
        plt.close(fig)

    return frames