import torch
import numpy as np
import copy
from utils import set_subtensor

def generate_true_state(x_test, idx=20):
    s = torch.Tensor(x_test[idx])
    return s, idx

def calc_mass_center(x_train):
    return np.mean(np.sum(x_train, axis=(1,2,3)))

class MuonPOMDP():
    def __init__(
            self,
            belief_updater,
            obs_surrogate,
            muon_data,
            muon_test_data,
            mass_center):
        self.initial_obs = torch.full((1, 1, 200, 200), -1.0)
        self.belief_updater = belief_updater
        self.obs_surrogate = obs_surrogate
        self.muon_data = muon_data
        self.muon_test_data = muon_test_data
        self.mass_center = mass_center
        self.drill_cost = 1.0
        self.actions = [(x, y) for x in range(10) for y in range(10)]
        self.stop_actions = ['go', 'nogo']
        self.discount = 0.99
        self.device = next(obs_surrogate.parameters()).device


class ImagePOMDP():
    def __init__(
            self,
            belief_updater,
            obs_surrogate,
            classifier,
            classification_samples,
            actions,
            correct_reward,
            incorrect_reward,
            data,
            test_data,
            image_size=28):
        self.initial_obs = torch.full((1, 1, image_size, image_size), -1.0)
        self.image_size = image_size
        self.belief_updater = belief_updater
        self.obs_surrogate = obs_surrogate
        self.classifier = classifier
        self.data = data
        self.test_data = test_data
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.actions = actions # [(x, y) for x in range(10) for y in range(10)]
        self.stop_action = 'stop'
        self.discount = 0.99
        self.device = next(belief_updater.parameters()).device


def discount(pomdp):
    return pomdp.discount


def actions(pomdp, s_or_b=None):
    A = pomdp.actions
    if s_or_b is None:
        return A
    else:
        return [a for a in A if a not in s_or_b.history]


class Belief():
    def __init__(self, belief, history=[]):
        self.belief = belief
        self.history = history

    def clear_history(self):
        self.history = []


def initialize_belief(pomdp):
    with torch.no_grad():
        model = pomdp.belief_updater
        o = pomdp.initial_obs.to(pomdp.device)
        b = model.update(o)
        return Belief(b)


def rand(pomdp, b, m):
    with torch.no_grad():
        return State(pomdp.belief_updater.sample(b.belief, m=m), history=copy.copy(b.history))


def update(pomdp, b, a, o):
    bp_batch = pomdp.belief_updater.update(o)
    history = copy.deepcopy(b.history)
    if a is not None:
        history.append(a)
    return Belief(bp_batch, history)


class State():
    def __init__(self, state, idx=None, is_test=True, history=[]):
        self.state = state
        self.idx = idx
        self.is_test = is_test
        self.history = history

    def clear_history(self):
        self.history = []


def transition(pomdp, s, a):
    sp = copy.deepcopy(s)
    sp.history.append(a)
    return sp # No state transition dynamics


def observation(pomdp, a, sp):
    device = pomdp.device
    if sp.idx and not isinstance(pomdp, ImagePOMDP):
        # Get observation from precomputed muon forward model
        if sp.is_test:
            M = pomdp.muon_test_data[sp.idx]
        else:
            M = pomdp.muon_data[sp.idx]
        o = np.flipud(M.transpose(0,2,1,3).reshape(1, 1, 200,200))
        o = torch.from_numpy(o.copy()).float()
    else:
        # Get observation from muon surrogate
        sp_batch = sp.state.to(device)
        if sp_batch.ndim != 4:
            sp_batch = sp_batch.unsqueeze(0)
        o = pomdp.obs_surrogate(sp_batch)
    if a is not None: # None indicates full observation
        m = sp.state.shape[0]
        o_partial = pomdp.initial_obs.clone().to(device)
        if o_partial.shape[0] != m:
            ones = [1]*(o_partial.ndim-1)
            o_partial = o_partial.repeat(m, *ones)
        for prev_a in sp.history:
            x, y = prev_a
            o_partial = set_subtensor(o_partial, o, y, x)
        o = o_partial
    return o.to(device)


def reward(pomdp, s, a=None, dim=None):
    if isinstance(pomdp, MuonPOMDP):
        if a == "go" or a is None:
            return torch.sum(s.state, dim=dim).detach().cpu() - pomdp.mass_center
        elif a == "nogo":
            return 0.0
        else:
            return -pomdp.drill_cost
    elif isinstance(pomdp, ImagePOMDP):
        if a == pomdp.stop_action:
            truth_label = pomdp.data[s.state.idx][1]
            predicted_label = None
            return 0.0


def go_reward(pomdp, S):
    return torch.sum(S, dim=[1,2,3]).mean().detach().cpu() - pomdp.mass_center


def correct_action(pomdp, s):
    return "go" if reward(pomdp, s) > 0 else "nogo"


def belief_error(b, s, mae=True):
    if mae:
        return torch.mean(torch.abs(torch.mean(b, dim=0).detach().cpu() - s.cpu())).item() # mean absolute error
    else:
        return torch.mean(torch.pow(torch.mean(b, dim=0).detach().cpu() - s.cpu(), 2)).item() # mean squared error


def volume_error(b, s):
    belief_mean_volume = torch.sum(b, dim=[1,2,3]).mean().detach().cpu()
    state_volume = torch.sum(s).detach().cpu()
    return torch.abs(belief_mean_volume - state_volume).item()


def batched_belief_transition(pomdp, b, a, m=100, k=1):
    if b.belief[0].shape[0] == m:
        m = 1
    𝐬 = rand(pomdp, b, m=m)
    𝐬𝐩 = transition(pomdp, 𝐬, a)
    𝐨 = observation(pomdp, a, 𝐬𝐩)
    𝐛𝐩 = update(pomdp, b, a, 𝐨)
    return 𝐛𝐩
