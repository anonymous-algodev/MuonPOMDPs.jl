import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from torch.distributions import Normal
from scipy.integrate import quad
from pomdp import MuonPOMDP, State, rand, initialize_belief, update, actions, transition, observation, discount, reward, go_reward, belief_error, volume_error, correct_action
from particle_filter import ParticleFilter
from utils import set_subtensor, get_device, seeding, clear_gpu, green_text, red_text
import random
import copy
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor

def is_particle_filter(belief_updater):
    return type(belief_updater).__name__ == "ParticleFilter"


def compute_returns(R, γ=1.0):
    T = len(R)
    G = torch.zeros((T,))
    next_return = 0
    for t in reversed(range(T)):
        G[t] = R[t] + γ*next_return
        next_return = G[t]
    return G


def entropy(particles):
    p = particles.mean(dim=0)
    p = p.clamp(min=1e-6, max=1-1e-6)
    entropy = -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)
    return entropy.mean().item()


def batched_entropy(particles, weights):
    # batch_size, k, m, D = particles.shape
    p = particles.mean(dim=2)
    p = p.clamp(min=1e-6, max=1-1e-6)
    entropy = -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)
    # Average over state dimensions (D)
    entropy = entropy.mean(dim=2)
    # Compute weighted entropy across k belief samples
    weighted_entropy = (entropy * weights).sum(dim=1)
    return weighted_entropy


def compute_kl_divergence(b, bp, weights):
    # Compute mean probability estimates over the second dimension (axis=1)
    P = bp.mean(dim=2)
    Q = b.mean(dim=2)

    eps = 1e-8
    P = P.clamp(min=eps, max=1-eps)
    Q = Q.clamp(min=eps, max=1-eps)

    # Compute KL divergence per batch index
    kl_elementwise = P * (torch.log(P) - torch.log(Q))
    kl_summed = kl_elementwise.sum(dim=-1)
    kl_weighted = (kl_summed * weights).sum(dim=-1)

    return kl_weighted


def gaussian_log_likelihood(z, mean, log_var):
    d = z.shape[-1]
    log_term = torch.log(torch.tensor(2 * torch.pi, device=z.device, dtype=z.dtype))
    log_likelihood = -0.5 * (log_term + log_var + (z - mean) ** 2 / log_var.exp())
    return log_likelihood.sum(dim=-1)  # Shape: [k]


def particle_filter_update(filter, obs):
    return filter.update(obs)


def batched_sample(model, obs_or_belief, m=1, is_cvae=False):
    if is_particle_filter(model):
        sampled_states = []
        batch_size = obs_or_belief.size(0)
        filters = [copy.deepcopy(model) for _ in range(batch_size)]
        for i in range(batch_size):
            obs = obs_or_belief[i]
            states = filters[i].update(obs)
            sampled_states.append(states)
        sampled_states = torch.stack(sampled_states, dim=0)
    else:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            if is_cvae:
                batch_size = obs_or_belief.size(0)
                obs_enc = model.obs_encoder(obs_or_belief)
                obs_enc = obs_enc.repeat(m, 1)
                z = torch.randn(m * batch_size, 2 * model.latent_dim).to(device)
                zso = torch.cat([z, obs_enc], dim=1)
                sampled_states = model.decoder(zso)
            else:
                if isinstance(obs_or_belief, tuple):
                    obs_h, qo_mean, qo_log_var = obs_or_belief
                    obs_h = obs_h.to(device)
                    qo_mean = qo_mean.to(device)
                    qo_log_var = qo_log_var.to(device)
                else:
                    obs = obs_or_belief.to(device)
                    batch_size = obs.size(0)
                    obs_h, qo_mean, qo_log_var = model.obs_encoder(obs)

                # Sample z from the prior
                std = torch.exp(0.5 * qo_log_var)
                eps = torch.randn(batch_size, m, std.size(1), device=device)
                z = qo_mean.unsqueeze(1) + eps * std.unsqueeze(1)
                z = z.view(batch_size * m, -1)
                obs_h = obs_h.repeat(m, 1)

                # Generate state samples
                sampled_states = model.decoder(z, obs_h)
                sampled_states = sampled_states.view(batch_size, m, *sampled_states.shape[1:])  # Reshape back

    return sampled_states


def misfit(obs_samples, o, return_distr=False):
    o_mask = (o != -1)
    mask_denom = torch.sum(o_mask.int()) / (20*20)
    misfit_distr = (torch.pow((obs_samples * o_mask) - (o * o_mask), 2).mean(dim=(1,2,3)) / mask_denom).detach().cpu()
    mean_misfit = misfit_distr.mean().item()
    if return_distr:
        return mean_misfit, misfit_distr
    else:
        return mean_misfit


def kde_cdf(kde, x):
    pdf = lambda x: kde.evaluate(x)
    cdf_value, _ = quad(pdf, -np.inf, x)
    return cdf_value


def compute_volume_probability(pomdp, 𝐒, use_kde=True, use_normal=False, ε=1e-10):
    massive_ore = torch.sum(𝐒, dim=[1, 2, 3]).detach().cpu() - pomdp.mass_center
    if use_kde:
        # Kernel density estimation
        kde = gaussian_kde(massive_ore)
        p_zero = kde_cdf(kde, 0)
    elif use_normal:
        # Assuming Gaussian distribution
        mean_vol = massive_ore.mean()
        std_vol = massive_ore.std(unbiased=True) + ε
        normal_dist = Normal(mean_vol, std_vol)
        p_zero = normal_dist.cdf(torch.tensor(0)).item()
    else:
        # Empirical cdf
        sorted_data = torch.sort(massive_ore).values
        less_than_zero = sorted_data[sorted_data < 0]
        p_zero = len(less_than_zero) / len(sorted_data)
    return p_zero


def action_lookahead(
    pomdp, b, S,
    k=1, # number of beliefs to expand
    nsteps=1,
    gamma=0.95,
    m=100,
    use_kde=False,
    use_normal=True,
    delta=0.99,
    action_info=False,
    seed=0,
    is_cvae=False,
    update_belief=lambda pomdp, b, a, O, seed=0: (_ for _ in ()).throw(ValueError("Please implement `update_belief`"))
):
    if action_info:
        info = {}

    # 1) Compute p_zero, same as your code
    p_zero = compute_volume_probability(pomdp, S, use_kde=use_kde, use_normal=use_normal)
    if action_info:
        info['probability'] = p_zero

    # 2) If p_zero or (1 - p_zero) is above threshold, pick 'nogo' or 'go'.
    #    These are terminal => 0 future gain
    if p_zero >= delta:
        # Terminal
        if action_info:
            return "nogo", 0.0, info
        else:
            return "nogo", 0.0

    if 1 - p_zero >= delta:
        # Terminal
        # r = reward(pomdp, state, "go")
        r = go_reward(pomdp, S)
        if action_info:
            return "go", r, info
        else:
            return "go", r

    # 3) Check if no more actions exist
    actions_to_process = actions(pomdp, b)
    batch_size = len(actions_to_process)
    if batch_size == 0:
        # Must choose final 'go' or 'nogo'
        final_act = "go" if p_zero < 0.5 else "nogo"
        if final_act == "go":
            r = go_reward(pomdp, S)
        else:
            r = 0.0
        if action_info:
            return final_act, r, info
        else:
            return final_act, r

    # 4) Single pass to compute immediate gains for all candidate actions
    device = pomdp.device

    _O = pomdp.initial_obs.clone().to(device).repeat(k, 1, 1, 1)

    if isinstance(b.belief, tuple):
        obs_h, qo_mean, qo_log_var = b.belief
        obs_h = obs_h.repeat(k, 1)
        qo_mean = qo_mean.repeat(k, 1)
        qo_log_var = qo_log_var.repeat(k, 1)

        z = pomdp.belief_updater.reparameterize(qo_mean, qo_log_var)
        weights = 1
        recon_s = pomdp.belief_updater.decoder(z, obs_h)
    else:
        weights = 1
        recon_s = pomdp.belief_updater.sample(b.belief, m=k)

    Op_full = pomdp.obs_surrogate(recon_s)

    for hx, hy in b.history:
        _O = set_subtensor(_O, Op_full, hy, hx)


    # build batched observation for each candidate action
    _O_batch = _O.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    for i, (ax, ay) in enumerate(actions_to_process):
        _O_batch[i] = set_subtensor(_O_batch[i], Op_full, ay, ax)

    _O_batch = _O_batch.reshape(-1, _O_batch.shape[-3], _O_batch.shape[-2], _O_batch.shape[-1])

    # sample states from each updated observation
    _Sp_batch = batched_sample(pomdp.belief_updater, _O_batch, m=m, is_cvae=is_cvae)
    _Sp_batch_flat = _Sp_batch.view(batch_size, k, m, -1)
    S_flat = S.view(1, 1, m, -1)

    H_Sp = batched_entropy(_Sp_batch_flat, weights)
    H_S = batched_entropy(S_flat, 1)
    immediate_gains = H_S.item() - H_Sp.cpu().numpy()

    # 5) If nsteps=1, pick best immediate
    if nsteps == 1:
        max_idx = np.argmax(immediate_gains)
        best_v = immediate_gains[max_idx]
        best_a = actions_to_process[max_idx]
        if action_info:
            info['observations'] = _O
            return best_a, best_v, info
        else:
            return best_a, best_v

    # 6) If nsteps>1, we do recursion
    best_action = None
    best_value = -np.inf

    # for each action, compute immediate gain + gamma * future value
    for i, a_candidate in enumerate(actions_to_process):
        # immediate information gain
        information_gain = immediate_gains[i]

        # Mean of m states
        hallucinated_state = rand(pomdp, b, m=100)
        hallucinated_state.state = torch.mean(hallucinated_state.state, dim=0).unsqueeze(0)

        O = observation(pomdp, None, hallucinated_state)
        b_next, S_next, _ = update_belief(pomdp, b, a_candidate, O, seed=seed)

        # Recursively call `action_lookahead`
        # We don't care about the next action now, just the value
        _, future_v = action_lookahead(
            pomdp, b_next, S_next,
            k=k,
            nsteps=nsteps - 1,
            gamma=gamma,
            m=m,
            use_kde=use_kde,
            use_normal=use_normal,
            delta=delta,
            action_info=False,
            update_belief=update_belief,
            seed=seed,
            is_cvae=is_cvae,
        )

        Q_val = information_gain + gamma * future_v
        if Q_val > best_value:
            best_value = Q_val
            best_action = a_candidate

    if action_info:
        info['observations'] = _O
        return best_action, best_value, info
    else:
        return best_action, best_value


def action_oracle(pomdp, s, 𝐒, m=100, k=1, use_kde=True, use_normal=False, delta=0.99, action_info=False, is_cvae=False):
    if action_info:
        info = {}

    # Collect actions to process
    actions_to_process = actions(pomdp, s)
    batch_size = len(actions_to_process)

    p_zero = compute_volume_probability(pomdp, 𝐒, use_kde=use_kde, use_normal=use_normal)

    if action_info:
        info['probability'] = p_zero

    if p_zero >= delta:
        a = "nogo"
    elif 1 - p_zero >= delta:
        a = "go"
    elif batch_size == 0:
        # No more drill actions, make a final decision.
        a = "go" if p_zero < 0.5 else "nogo"
    else:
        # Initialize _O
        device = pomdp.device
        _O = pomdp.initial_obs.clone().to(device)
        Op_full = pomdp.obs_surrogate(s.state.unsqueeze(0))

        # Update _O with previous actions
        for prev_a in s.history:
            x, y = prev_a
            _O = set_subtensor(_O, Op_full, y, x)

        # Create batch of observations
        # _O_batch = (|A| actions, k beliefs, C, W, H)
        _O_batch = _O.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        # Update observations for each action
        for (i, a) in enumerate(actions_to_process):
            _O_batch[i] = set_subtensor(_O_batch[i], Op_full, a[1], a[0])

        _O_batch = _O_batch.reshape(-1, _O_batch.shape[-3], _O_batch.shape[-2], _O_batch.shape[-1])
        _Sp_batch = batched_sample(pomdp.belief_updater, _O_batch, m=m, is_cvae=is_cvae)  # Shape: (batch_size, m, state_channels, 80, 80)
        _Sp_batch = _Sp_batch.reshape(-1, k, m, _Sp_batch.shape[-3], _Sp_batch.shape[-2], _Sp_batch.shape[-1])

        # Compute the mean belief over m samples
        mean_S = _Sp_batch.mean(dim=2)

        # Repeat true state to match batches
        s_batch = s.state.repeat(mean_S.shape[0], k, 1, 1, 1)

        # Ensure the predictions are in [0, 1]
        mean_S = torch.sigmoid(mean_S)

        loss = F.l1_loss(mean_S, s_batch, reduction='none')
        loss = loss.mean(dim=(2,3,4))
        loss = loss.mean(dim=1)
        min_idx = torch.argmin(loss)

        a = actions_to_process[min_idx]

        if action_info:
            info['observations'] = _O

    if action_info:
        return a, info
    else:
        return a


def action_rand(pomdp, b, 𝐒, use_kde=True, use_normal=False, fully_random=False, delta=0.99, action_info=False, is_cvae=False):
    if action_info:
        info = {}

    # Collect actions to process
    actions_to_process = actions(pomdp, b)
    batch_size = len(actions_to_process)

    p_zero = compute_volume_probability(pomdp, 𝐒, use_kde=use_kde, use_normal=use_normal)

    if action_info:
        info['probability'] = p_zero

    if not fully_random and p_zero >= delta:
        a = "nogo"
    elif not fully_random and 1 - p_zero >= delta:
        a = "go"
    elif batch_size == 0:
        # No more drill actions, make a final decision.
        a = "go" if p_zero < 0.5 else "nogo"
    else:
        if fully_random:
            all_actions = copy.copy(actions_to_process)
            all_actions.append("go")
            all_actions.append("nogo")
        else:
            all_actions = actions_to_process
        a = random.choice(all_actions)

        if action_info:
            device = pomdp.device
            if is_particle_filter(pomdp.belief_updater):
                recon_s = torch.mean(b.belief, dim=0).unsqueeze(0)
            elif is_cvae:
                batch_size = b.belief.size(0)
                obs_enc = pomdp.belief_updater.obs_encoder(b.belief)
                z = torch.randn(batch_size, 2 * pomdp.belief_updater.latent_dim).to(device)
                zso = torch.cat([z, obs_enc], dim=1)
                recon_s = pomdp.belief_updater.decoder(zso)
            else:
                z = pomdp.belief_updater.reparameterize(b.belief[1], b.belief[2])
                recon_s = pomdp.belief_updater.decoder(z, b.belief[0])
            Op_full = pomdp.obs_surrogate(recon_s)  # Shape: (1, C, W, H)

            _O = pomdp.initial_obs.clone().to(device)

            # Update _O with previous actions
            for prev_a in b.history:
                x, y = prev_a
                _O = set_subtensor(_O, Op_full, y, x)

            info['observations'] = _O

    if action_info:
        return a, info
    else:
        return a


def create_left_right_grid():
    pattern = []
    # y = 9 down to 0
    for row in range(9, -1, -1):
        # Determine if we go left -> right or right -> left
        row_index_from_top = 9 - row
        if row_index_from_top % 2 == 0:
            # even row index from the top: left -> right
            for col in range(0, 10):
                pattern.append((col, row))
        else:
            # odd row index from the top: right -> left
            for col in range(9, -1, -1):
                pattern.append((col, row))
    return pattern


def create_up_down_grid():
    pattern = []
    for x in range(10):
        if x % 2 == 0:
            # Even column x: go bottom -> top (y=0..9)
            for y in range(10):
                pattern.append((x, y))
        else:
            # Odd column x: go top -> bottom (y=9..0)
            for y in range(9, -1, -1):
                pattern.append((x, y))
    return pattern


def action_grid(pomdp, b, 𝐒, use_kde=True, use_normal=False,
                delta=0.99, action_info=False, grid_type='vertical', is_cvae=False):
    if action_info:
        info = {}

    # Collect actions to process
    actions_to_process = actions(pomdp, b)
    batch_size = len(actions_to_process)

    p_zero = compute_volume_probability(pomdp, 𝐒,
                                        use_kde=use_kde,
                                        use_normal=use_normal)
    if action_info:
        info['probability'] = p_zero

    # Decision thresholds based on p_zero
    if (p_zero >= delta):
        a = "nogo"
    elif (1 - p_zero >= delta):
        a = "go"
    elif batch_size == 0:
        # No more drilling actions, make a final decision
        a = "go" if p_zero < 0.5 else "nogo"
    else:
        if grid_type == 'horizontal':
            grid_pattern = create_left_right_grid()
        else:
            grid_pattern = create_up_down_grid()
        # Pick only actions that are still valid
        valid_actions = [pt for pt in grid_pattern if pt in actions_to_process]

        if valid_actions:
            a = valid_actions[0]
        else:
            raise Exception("Error: No more actions available in grid pattern.")

        if action_info:
            device = pomdp.device
            if is_particle_filter(pomdp.belief_updater):
                recon_s = torch.mean(b.belief, dim=0).unsqueeze(0)
            elif is_cvae:
                batch_size = b.belief.size(0)
                obs_enc = pomdp.belief_updater.obs_encoder(b.belief)
                z = torch.randn(batch_size, 2 * pomdp.belief_updater.latent_dim).to(device)
                zso = torch.cat([z, obs_enc], dim=1)
                recon_s = pomdp.belief_updater.decoder(zso)
            else:
                z = pomdp.belief_updater.reparameterize(b.belief[1], b.belief[2])
                recon_s = pomdp.belief_updater.decoder(z, b.belief[0])
            Op_full = pomdp.obs_surrogate(recon_s)  # Shape: (1, C, W, H)

            _O = pomdp.initial_obs.clone().to(device)

            # Update _O with previous actions
            for prev_a in b.history:
                x, y = prev_a
                _O = set_subtensor(_O, Op_full, y, x)

            info['observations'] = _O

    if action_info:
        return a, info
    else:
        return a


def simulate(
        pomdp,
        state_obj,
        is_test=True,
        delta=0.99,
        m=100,
        k=1,
        seed=0,
        verbose=False,
        progress=False,
        rand_policy=False,
        fully_random=False,
        oracle_policy=False,
        grid_policy=False,
        grid_type='vertical',
        use_normal=True,
        use_kde=False,
        is_cvae=False,
        nsteps=1):
    
    # Set random seed for reproducibility
    seeding(seed)

    if is_particle_filter(pomdp.belief_updater):
        pomdp.belief_updater.reset()

    s = state_obj.state
    s_idx = state_obj.idx

    device = pomdp.device
    pomdp.belief_updater = pomdp.belief_updater.to(device)
    pomdp.obs_surrogate = pomdp.obs_surrogate.to(device)

    s = s.to(device)
    A = actions(pomdp) # TODO. Only drill actions.

    state = State(s, idx=s_idx, is_test=is_test)

    b = initialize_belief(pomdp)
    𝐒 = rand(pomdp, b, m=m).state
    O_true_full = observation(pomdp, None, state)

    R = []
    Rb = []

    info = {
        'b': [b],
        'beliefs': [𝐒],
        'errors': [belief_error(𝐒, s)],
        'volume_errors': [volume_error(𝐒, s)],
        'probabilities': [],
        'misfits': [],
        'returns': None,
        'obs': [pomdp.initial_obs.clone().to(device)],
        'times': [],
    }

    def update_belief(pomdp, b, a, O, seed=seed):
        # Closures: device, m
        o = pomdp.initial_obs.clone().to(device)
        for prev_a in b.history + [a]:
            x, y = prev_a
            o = set_subtensor(o, O, y, x)

        # Update belief
        seeding(seed)
        bp = update(pomdp, b, a, o)
        𝐒p = rand(pomdp, bp, m=m).state
        return (bp, 𝐒p, o)

    true_massive_ore = reward(pomdp, state)
    correct_act = correct_action(pomdp, state)

    with torch.no_grad():
        # T = range(1) if delta == np.inf else range(1, 102)
        T = range(1, 102)
        T = tqdm(T) if progress else T

        for t in T:
            stime = time.time()

            ## Selection
            if oracle_policy:
                a, act_info = action_oracle(pomdp, state, 𝐒, k=k, m=m, use_kde=use_kde, use_normal=use_normal, delta=delta, action_info=True, is_cvae=is_cvae)
            elif rand_policy:
                a, act_info = action_rand(pomdp, b, 𝐒, use_kde=use_kde, use_normal=use_normal, delta=delta, fully_random=fully_random, action_info=True, is_cvae=is_cvae)
            elif grid_policy:
                a, act_info = action_grid(pomdp, b, 𝐒, use_kde=use_kde, use_normal=use_normal, delta=delta, grid_type=grid_type, action_info=True, is_cvae=is_cvae)
            else:
                a, _, act_info = action_lookahead(pomdp, b, 𝐒, k=k, nsteps=nsteps, m=m, use_kde=use_kde, use_normal=use_normal, delta=delta, action_info=True, seed=seed, update_belief=update_belief, is_cvae=is_cvae)

            info['probabilities'].append(act_info['probability'])

            r = reward(pomdp, state, a)
            R.append(r)

            if verbose:
                print(f"Time = {t}, Action = {a}, P(v < 0) ≈ {info['probabilities'][-1]}, r = {r}")

            if isinstance(a, str):
                break
            else:
                bp, 𝐒p, o = update_belief(pomdp, b, a, O_true_full, seed=t*seed)

                info['b'].append(bp)
                info['beliefs'].append(𝐒p)
                info['obs'].append(o)
                info['errors'].append(belief_error(𝐒p, s))
                info['volume_errors'].append(volume_error(𝐒p, s))
                info['misfits'].append(misfit(act_info['observations'], O_true_full))

                etime = time.time()
                elapsed = etime - stime
                info['times'].append(elapsed)

                b = bp
                𝐒 = 𝐒p

            ## Expansion
            state = transition(pomdp, state, a)

    info['is_correct'] = a == correct_act
    info['returns'] = compute_returns(R, γ=discount(pomdp))
    info['discounted_return'] = info['returns'][0].item()
    info['action_history'] = b.history

    if verbose:
        est_massive_ore_mean = torch.mean(torch.sum(𝐒, dim=[1,2,3]) - pomdp.mass_center)
        est_massive_ore_std = torch.std(torch.sum(𝐒, dim=[1,2,3]) - pomdp.mass_center)
        indicator_text = green_text("True") if a == correct_act else red_text("False")
        print(f"Correct ? {indicator_text} (Truth = {true_massive_ore:.5g}, Est. = {est_massive_ore_mean:.5f} ± {est_massive_ore_std:.5g})")

    clear_gpu()
    return info
