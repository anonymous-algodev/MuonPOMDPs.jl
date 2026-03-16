import torch
from torch.utils.data import Dataset
import numpy as np
import random
from random import randint

def split_index_every(index, every=100):
    i = index // every  # Integer division for i
    j = index % every + 1   # Remainder for j
    return i, j

class StateDataset(Dataset):
    def __init__(
            self,
            data,
            data3d,
            muon_data,      # e.g., PRECOMPUTED_MUON
            muon_test_data, # e.g., PRECOMPUTED_MUON_TEST
            num_samples,
            image_size=80,
            num_observations=10,
            grid=True,
            step=10,
            is_muon=True,
            rand_obs=True,
            all_obs=False,
            force_num_obs=False,
            is_test=False,
            return_full_obs=False,
            return_actions=False):
        """
        Initialize the dataset.

        Args:
            num_samples (int): Number of samples in the dataset.
            image_size (int): Size of the state image (image_size x image_size).
            num_observations (int): Number of observations per state image.
        """
        self.data = data
        self.data3d = data3d
        self.muon_data = muon_data
        self.muon_test_data = muon_test_data
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_observations = num_observations
        self.grid = grid
        self.step = step
        self.rand_obs = rand_obs
        self.all_obs = all_obs
        self.is_muon = is_muon
        self.force_num_obs = force_num_obs
        self.is_test = is_test
        self.return_full_obs = return_full_obs
        self.return_actions = return_actions

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx, use_index=False):
        # Initialize the observation history map with -1 (unobserved)
        if self.is_muon:
            condition = np.full((1, 10, 10, 20, 20), -1.0)
        else:
            condition = torch.full((1, self.image_size, self.image_size), -1.0)

        if self.force_num_obs:
            if use_index:
                i = idx
            else:
                i = randint(0, len(self.data)-1)
            num_obs = self.num_observations
        elif self.rand_obs:
            i = randint(0, len(self.data)-1)
            num_obs = idx % (self.num_observations+1)
        else:
            i, num_obs = split_index_every(idx)

        if self.all_obs:
            num_obs = self.num_observations

        state = torch.Tensor(self.data[i])
    
        # Simulate observations
        if self.is_muon:
            if self.is_test:
                M = self.muon_test_data[i]
            else:
                M = self.muon_data[i]

        actions = []
        if self.rand_obs:
            if self.is_muon:
                xy = [(x,y) for x in range(10) for y in range(10)]
                rand_subset = random.sample(xy, num_obs)
                for (x,y) in rand_subset:
                    condition[0, y, x] = M[y,x] # NOTE. y,x flip
                    actions.append((x,y))
            else:
                if self.grid:
                    xy = [(x,y) for x in range(4,80,8) for y in range(4,80,8)]
                else:
                    xy = [(x,y) for x in range(self.image_size) for y in range(self.image_size)]
                rand_subset = random.sample(xy, num_obs)
                for (x,y) in rand_subset:
                    value = state[0, x, y].item()
                    condition[0, x, y] = value
                    actions.append((x,y))
        else:
            if self.is_muon:
                X = range(10)
                Y = range(10)
            else:
                if self.grid:
                    X = range(4, self.image_size, 8)
                    Y = range(4, self.image_size, 8)
                else:
                    X = range(self.image_size)
                    Y = range(self.image_size)
            count = 0
            for x in X:
                for y in Y:
                    count += 1
                    if self.is_muon:
                        condition[0, y, x] = M[y,x] # NOTE. y,x flip
                    else:
                        value = state[0, x, y].item()
                        condition[0, x, y] = value
                    if count >= num_obs:
                        break
                if count >= num_obs:
                    break

        if self.is_muon:
            condition = np.flipud(condition.reshape(10,10,20,20).transpose(0,2,1,3).reshape(1, 200,200))
            condition = torch.from_numpy(condition.copy()).float()

        if self.return_full_obs:
            full_obs = np.flipud(M.reshape(10,10,20,20).transpose(0,2,1,3).reshape(1, 200,200))
            full_obs = torch.from_numpy(full_obs.copy()).float()
            if self.return_actions:
                return state, condition, full_obs, torch.tensor(actions)
            else:
                return state, condition, full_obs
        else:
            if self.return_actions:
                return state, condition, torch.tensor(actions)
            else:
                return state, condition
