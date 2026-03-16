from numbers import Number
import numpy as np

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


