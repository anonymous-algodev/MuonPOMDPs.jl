import MuonForward.src.Tomo3D as MuonTomo
from MuonForward.src.utils import *
from discretize import TensorMesh
from SimPEG import maps
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import requests
import io
import numpy as np
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "data", "muon_sim_simpeg.pkl"))

def save_sim(obj, filename=SIM_DATA_PATH):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_sim(filename=SIM_DATA_PATH):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def normalize(ordered_dict, is_global=True):
    if is_global:
        # Find the global minimum and maximum across all arrays
        all_values = np.concatenate(list(ordered_dict.values()))
        global_min = np.min(all_values)
        global_max = np.max(all_values)

        # Avoid division by zero if global_max == global_min
        normalized_dict = OrderedDict()
        if global_max != global_min:
            for key, array in ordered_dict.items():
                normalized_array = (array - global_min) / (global_max - global_min)
                normalized_dict[key] = normalized_array
        else:
            # If all values are the same, return arrays of zeros
            for key, array in ordered_dict.items():
                normalized_dict[key] = np.zeros_like(array)

        return normalized_dict
    else:
        normalized_dict = OrderedDict()
        
        for key, array in ordered_dict.items():
            min_val = np.min(array)
            max_val = np.max(array)
            
            # Avoid division by zero if max_val == min_val
            if max_val != min_val:
                normalized_array = (array - min_val) / (max_val - min_val)
            else:
                # If all values are the same, the normalized values will be 0
                normalized_array = np.zeros_like(array)
            
            normalized_dict[key] = normalized_array
        
        return normalized_dict


def divide_ordered_dicts(dict1, dict2, convert_nans=True):
    result_dict = OrderedDict()

    for key in dict1.keys():
        # Element-wise division of corresponding arrays
        result_dict[key] = dict1[key] / dict2[key]
        if convert_nans:
            result_dict[key] = np.nan_to_num(result_dict[key])

    return result_dict


def mae_ordered_dicts(dict1, dict2):
    result_dict = OrderedDict()

    for key in dict1.keys():
        # Element-wise MAE of corresponding arrays
        result_dict[key] = np.abs(dict1[key] - dict2[key])

    return result_dict


def mse_ordered_dicts(dict1, dict2):
    result_dict = OrderedDict()

    for key in dict1.keys():
        # Element-wise MAE of corresponding arrays
        result_dict[key] = (dict1[key] - dict2[key])**2

    return result_dict


def generate_muon(state, topography, muon_sim_simpeg=None, rerun=False, avg_density=False, is_norm=False):
    # density model, with intrusion density = 1.5, background density 1 and air density 0
    dens_intr = 1.5 # intrusive rock
    dens_hst = 1.0  # host rock

    # Run forward model on the geological model
    model = state.copy()
    dens_m = np.zeros_like(model) # create density model
    dens_m[model==1] = dens_intr
    dens_m[model==0] = dens_hst

    # create mask for topography
    topo_mask = state.copy()
    topo_mask[topo_mask==0] = True
    topo_mask[topo_mask==1] = True

    # dimension of the mesh
    x_dim, y_dim, z_dim = state.shape
    n = (x_dim, y_dim, z_dim)

    # scale of the mesh
    hx, hy, hz = 50.0, 50.0, 10.0 # meters
    h = np.array([hx, hy, hz])
    x0 = np.array([0., 0., -480.])
    hole_nx, hole_ny = 10, 10

    xbh, ybh = np.meshgrid(np.linspace(200,3800, hole_nx),
                           np.linspace(200,3800, hole_ny))
    xbh, ybh = xbh.flatten(), ybh.flatten()
    zbh = np.array([snap_to_topo((xbh[i], ybh[i],0), topography, x0, h, n)[2] for i in range(len(xbh))])

    dips = 180.0 + np.full_like(zbh, 60.0)
    azimuth = 90.0
    lengths = np.repeat(1100, hole_nx*hole_ny) # borehole length

    bhs = [Borehole((xbh[i], ybh[i], zbh[i]), dips[i], azimuth, lengths[i], i) for i in range(len(xbh))]

    # Create a discretize tensor mesh
    mesh = TensorMesh([[hi] * ni for (hi,ni) in zip(h,n)], x0=np.array([0,0,-480.0]))

    # Create radiograph grid specifying the grid of ray directions
    # along which opacities will be computed.
    tan_theta_max = np.tan(np.radians(60))
    nx = 20
    ny = 20

    if rerun:
        xgrid = np.linspace(-tan_theta_max, tan_theta_max, nx)
        ygrid = np.linspace(-tan_theta_max, tan_theta_max, ny)
        # Create muon sensors
        sensors = OrderedDict()

    n_sensors_per_bh = 9
    sensor_locs = np.zeros((len(bhs)*n_sensors_per_bh, 3))
    for (ibh, bh) in enumerate(bhs):
        for (i,ti) in enumerate(np.linspace(150.0, lengths[ibh]-2.5*h[2], n_sensors_per_bh)):
            loc = bh(ti)
            sensor_locs[bh.id*n_sensors_per_bh+i] = loc
            if rerun:
                sensors[f'{bh.id}_{i}'] = MuonTomo.MuonSensor(loc, xgrid, ygrid)

    # Define active cell mapping, for later use in SimPEG inversion
    # SimPEG inversion will work on all cells above the lowest sensor and
    # below the topography. Forward operator will be applied to all cells
    # after mapping the active cell model to the full mesh.
    minz = np.min(sensor_locs[:,2])
    gridCC = mesh.gridCC
    is_active = np.logical_not(np.isnan(topo_mask))
    is_active = is_active.flatten(order='F')
    is_active = np.logical_and(is_active, gridCC[:,2] >= minz - h[2])

    if rerun:
        print('It will take around 5-20min. Once need to run once')
        active_map = maps.InjectActiveCells(mesh, is_active, valInactive=0.0)
        muon_sim_simpeg = MuonTomo.ToyMuonSimulationSimPeg(mesh, sensors, model_map=active_map)
        save_sim(muon_sim_simpeg)
    elif muon_sim_simpeg is None:
        raise Exception("muon_sim_simpeg is None, please pass it in via `load_sim()`")

    # Run forward model to get muon sensor data
    d_muon = muon_sim_simpeg.get_data(dens_m.flatten(order='F')[is_active])

    if avg_density:
        # Null model (no intrusion) with density of 1 for demonstration purposes
        rho0 = np.ones(mesh.nC)

        # Forward model the null model
        d0_muon = muon_sim_simpeg.get_data(rho0[is_active])

        if is_norm:
            return divide_ordered_dicts(normalize(d_muon), normalize(d0_muon))
        else:
            return divide_ordered_dicts(d_muon, d0_muon)
    else:
        return normalize(d_muon) if is_norm else d_muon


def plot_radiographs(d_muon):
    # Plot null model radiographs: (null = empty space, only topography without intrusion inside)
    nx = 20
    ny = 20
    plt.subplots(9,10,figsize=(30, 20))
    c = 0
    for j in range(0, 9):
        for i in range(60, 70):
        # for i in range(5, 100, 10):
            c += 1
            data_id = f"{i}_{j}"
            data_arr = d_muon[data_id]
            plt.subplot(9,10,c)
            ax = plt.gca()
            img = ax.imshow(data_arr.reshape(nx, ny).T, cmap="viridis", origin="lower") # NOTE: .T transpose is important
            ax.set_title(f"Radiograph, Sensor {data_id}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.axis('off')
            cbar = plt.colorbar(img, ax=ax)
            cbar.ax.tick_params(labelsize=8)

    plt.show()


def plot_radiographs_all(d_muon, top_lvl=False, sliced=False, vmin=None, vmax=None):
    # Plot null model radiographs: (null = empty space, only topography without intrusion inside)
    nx = 20
    ny = 20
    if sliced:
        plt.subplots(1,10,figsize=(30, 20))
        xy = range(60, 70)
    else:
        plt.subplots(10,10,figsize=(30, 20))
        xy = [num for i in range(90, -1, -10) for num in range(i, i + 10)]
    for i,ri in enumerate(xy):
        if top_lvl:
            m = []
        else:
            m = np.zeros((nx, ny))
        for j in range(0, 9):
            data_id = f"{ri}_{j}"
            data_arr = d_muon[data_id]
            if top_lvl:
                m.append(data_arr.reshape(nx,ny))
            else:
                m += data_arr.reshape(nx,ny)
        if top_lvl:
            m = np.max(np.stack(m), axis=0)
        else:
            m /= 9
        if sliced:
            plt.subplot(1,10,i+1)
        else:
            plt.subplot(10,10,i+1)
        ax = plt.gca()
        if vmin is not None and vmax is not None:
            img = ax.imshow(m.T, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax) # NOTE: .T transpose is important
        else:
            img = ax.imshow(m.T, cmap="viridis", origin="lower") # NOTE: .T transpose is important
        ax.set_title(f"{ri}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off')
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    plt.show()


def get_muon_observation(muon, x, y):
    ax = ay = 10
    az = 9
    xy2ind = np.array(list(muon.keys())).reshape(ax,ay,az)
    keys = xy2ind[y,x,:] # NOTE: y,x flipped
    k = keys[0]
    m = np.zeros(muon[k].shape)
    nx = ny = int(np.sqrt(m.shape[0]))
    for k in keys:
        m += muon[k]
    m /= az
    return m.reshape(nx,ny).T # NOTE: Transpose is important


def normalize_array(array):
    # Find the global minimum and maximum
    global_min = np.min(array)
    global_max = np.max(array)
    if global_min == global_max:
        return np.zeros_like(array)
    else:
        return (array - global_min) / (global_max - global_min)
    

def combine_obs(d_muon, top_lvl=True, norm=True):
    # Plot null model radiographs: (null = empty space, only topography without intrusion inside)
    nx = 20
    ny = 20
    xy = [num for i in range(90, -1, -10) for num in range(i, i + 10)]
    M = 100*[None]
    for ri in xy:
        if top_lvl:
            m = []
        else:
            m = np.zeros((nx, ny))
        for j in range(0, 9):
            data_id = f"{ri}_{j}"
            data_arr = d_muon[data_id]
            data = data_arr.reshape(nx,ny)
            if top_lvl:
                m.append(data)
            else:
                m += data
        if top_lvl:
            m = np.max(np.stack(m), axis=0)
        else:
            m /= 9
        M[ri] = m.T # NOTE: .T transpose is important
    M = np.stack(M)
    if norm:
        return normalize_array(M)
    else:
        return M