import pickle
import requests
import io
import os
from tqdm import tqdm
import torch
import numpy as np
from observations import load_sim, generate_muon, combine_obs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MUON_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "data", "muon_maps.pkl"))
MUON_AUG_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "data", "muon_maps_aug.pkl"))
MUON_TEST_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "data", "muon_test_maps.pkl"))
MUON_TEST_AUG_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "data", "muon_test_maps_aug.pkl"))

def save_muons(obj, filename=MUON_DATA_PATH):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_muons(filename=MUON_DATA_PATH):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def precompute_muon_obs(
    obs_model,
    x_train, x_train3d,
    x_test, x_test3d,
    use_obs_surrogate=False,
    rerun_precompiled_muon=False,
    rerun_precompiled_muon_test=False,
    use_augmented=False
):
    # load topography
    response = requests.get('https://www.dropbox.com/scl/fi/lua5fwx9dwejml14u3bxp/topography.npy?rlkey=2mrl452k7u0ubn0vwzfp61gm6&dl=1')
    response.raise_for_status()
    topography = np.load(io.BytesIO(response.content)) * 10

    muon_sim_simpeg = load_sim()

    if use_augmented:
        muon_train_filename = MUON_AUG_DATA_PATH
    else:
        muon_train_filename = MUON_DATA_PATH

    if rerun_precompiled_muon:
        device1 = next(obs_model.parameters()).device
        precomputed_muon = []
        for i in tqdm(range(x_train.shape[0])):
            if use_obs_surrogate:
                with torch.no_grad():
                    M = obs_model(torch.Tensor(x_train[i]).unsqueeze(0).to(device1)).detach().cpu().numpy()
                    M = np.flipud(M).reshape(10, 20, 10, 20).transpose(0, 2, 1, 3)
                    # M = torch.Tensor(M) # Needs to be numpy array
            else:
                d_muon = generate_muon(x_train3d[i], topography, avg_density=True, is_norm=False, muon_sim_simpeg=muon_sim_simpeg)
                M = combine_obs(d_muon)
                M = M.reshape(10,10,20,20)
            precomputed_muon.append(M)
        save_muons(precomputed_muon, filename=muon_train_filename)
    else:
        precomputed_muon = load_muons(filename=muon_train_filename)

    if use_augmented:
        muon_test_filename = MUON_TEST_AUG_DATA_PATH
    else:
        muon_test_filename = MUON_TEST_DATA_PATH

    if rerun_precompiled_muon_test:
        device1 = next(obs_model.parameters()).device
        precomputed_muon_test = []
        for i in tqdm(range(x_test.shape[0])):
            if use_obs_surrogate:
                with torch.no_grad():
                    M = obs_model(torch.Tensor(x_test[i]).unsqueeze(0).to(device1)).detach().cpu().numpy()
                    M = np.flipud(M).reshape(10, 20, 10, 20).transpose(0, 2, 1, 3)
                    # M = torch.Tensor(M) # Needs to be numpy array
            else:
                d_muon = generate_muon(x_test3d[i], topography, avg_density=True, is_norm=False, muon_sim_simpeg=muon_sim_simpeg)
                M = combine_obs(d_muon)
                M = M.reshape(10,10,20,20)
            precomputed_muon_test.append(M)
        save_muons(precomputed_muon_test, filename=muon_test_filename)
    else:
        precomputed_muon_test = load_muons(filename=muon_test_filename)

    return precomputed_muon, precomputed_muon_test
