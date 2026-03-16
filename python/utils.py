from numbers import Number
import numpy as np
import torch
import random
import re
import json

def set_subtensor(t_original, t_new, grid_row, grid_col, grid_size=10, sub_size=20):
    # Validate grid indices
    if not (0 <= grid_row < grid_size) or not (0 <= grid_col < grid_size):
        raise IndexError(f"Grid indices out of range. Received grid_row={grid_row}, grid_col={grid_col}")

    # Calculate start and end indices
    row_start = grid_row * sub_size
    row_end = (grid_row + 1) * sub_size
    col_start = grid_col * sub_size
    col_end = (grid_col + 1) * sub_size

    # Extract the sub-tensor
    t_original = t_original.clone()
    t_original[:, :, row_start:row_end, col_start:col_end] = t_new[:, :, row_start:row_end, col_start:col_end]
    return t_original

def seeding(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def clear_gpu():
    torch.cuda.empty_cache()

def get_device(model):
    return next(model.parameters()).device

def green_text(str):
    return f"\033[1;32m{str}\033[0m"

def red_text(str):
    return f"\033[1;31m{str}\033[0m"

def custom_json_formatter(data):
    # Step 1: Pretty-print with indentation
    pretty_json = json.dumps(data, indent=4)
    
    # Step 2: Use regex to find arrays and flatten them to one line
    def flatten_array(match):
        array_content = match.group(1)
        # Remove newlines and excessive spaces inside arrays
        flattened_content = re.sub(r'\s*\n\s*', ' ', array_content.strip())
        return f'[{flattened_content}]'

    # Apply regex transformation to flatten arrays
    formatted_json = re.sub(r'\[\s*([\s\S]*?)\s*\]', flatten_array, pretty_json)

    return formatted_json

def saveresults(data, filename):
    formatted_json = custom_json_formatter(data)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(formatted_json)  # Write formatted JSON string to file

def loadresults(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def get_cell_index(p, x0, h, n):
    """
    Get cell index of a point on a 3D regular mesh will cell ordering
    reshapeable to (nx, ny, nz) with c-ordering.
    """
    ix = int(np.floor((p[0] - x0[0])/h[0]))
    assert ix >= 0 and ix < n[0], "x index out of bounds"
    iy = int(np.floor((p[1] - x0[1])/h[1]))
    assert iy >= 0 and iy < n[1], "y index out of bounds"
    iz = int(np.floor((p[2] - x0[2])/h[2]))
    assert iz >= 0 and iz < n[2], "z index out of bounds"
    return ix, iy, iz

def get_topo_index(p, topo, x0, h, n):
    """
    get vertical cell index of a point.
    """
    ix, iy, izp = get_cell_index(p, x0, h, n)
    assert (ix >= 0 and ix < n[0]) & (iy >= 0 and iy < n[1]), "point outside horizontal extents of mesh"
    iztopo = int(np.floor((topo[ix, iy] - x0[2])/h[2]))
    return iztopo

def snap_to_topo(p, topo, x0, h, n) -> np.ndarray:
    """
    Snap a point to the topography of a 3D regular mesh with cell ordering
    reshapeable to (nx, ny, nz) with c-ordering.
    """
    iz = get_topo_index(p, topo, x0, h, n)
    return np.array([p[0], p[1], x0[2]+iz*h[2]])

# Define a borehole as a ray extending downward from surface
class Ray(object):
    def __init__(self, x0, theta_d, phi_d) -> None:
        self.theta = np.radians(theta_d)
        self.phi = np.radians(phi_d)
        self.x0 = x0
        self.d = np.zeros(3)
        self.d[0] = np.sin(self.theta)*np.cos(self.phi)
        self.d[1] = np.sin(self.theta)*np.sin(self.phi)
        self.d[2] = np.cos(self.theta)

    @classmethod
    def from_rd(cls, x0, rd):
        rd = rd/np.linalg.norm(rd)
        theta = np.arccos(rd[2])
        phi = np.arctan2(rd[1], rd[0])
        return cls(x0, np.degrees(theta), np.degrees(phi))

    def __call__(self, t: Number) -> Number:
        assert t >= 0, "t must be non-negative"
        return self.x0 + t*self.d

class Borehole(object):
    def __init__(self, x0, theta_d, phi_d, length, id) -> None:
        self.ray = Ray(x0, theta_d, phi_d)
        self.length = length
        self.id = id
        self.x0 = x0

    def __call__(self, t: Number) -> Number:
        assert (t >= 0) & (t <= self.length), "t must be non-negative and <= borehole length"
        return self.ray(t)


