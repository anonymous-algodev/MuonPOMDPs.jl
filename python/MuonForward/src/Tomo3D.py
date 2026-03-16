from collections import OrderedDict
from typing import Dict, Any
import numpy as np
import numba

from SimPEG.simulation import LinearSimulation
import SimPEG.maps as maps
from SimPEG.utils import mkvc, sdiag
from SimPEG.survey import BaseSurvey
from discretize import TensorMesh
import scipy.sparse as sparse

class MuonSensor(object):
    """
    Toy muon tomography sensor class. Assume each muon measurement consists of the
    opacity (line integral of density along a raypath) for a single raypath. Each
    sensor contains measurements on a 2D cartesian tensor product grid of raypath
    directions. The raypath directions are specified by the tan(theta_x) and
    tan(theta_y) coordinates of the raypath direction, where theta_x and theta_y
    are the angles between vertical and the x and y components of the raypath
    direction, respectively.

    Constructor inputs:

    loc: np.ndarray -> location of the sensor as length 3 np.ndarray
    rgrid_x: np.ndarray -> 1D array of tan(theta_x) coordinates of raypath grid
    rgrid_y: np.ndarray -> 1D array of tan(theta_y) coordinates of raypath grid
    """
    def __init__(self,
        loc: np.ndarray,
        rgrid_x: np.ndarray,
        rgrid_y: np.ndarray
    ) -> None:
        self.loc = loc
        self.rgrid_x = rgrid_x
        self.rgrid_y = rgrid_y

class ToyMuonSimulationSimPeg(LinearSimulation):
    """
    SimPEG simulation class for toy muon tomography problem.
    Performs linear forward modelling and sensitivity calculations
    on a SimPEG TensorMesh.

    Linear forward operator is constructed as a scipy CSR sparse matrix G. This
    class provides the methods required to interface with SimPEG inversion
    routines.

    Note that following SimPEG conventions, the forward modelling methods in this
    class compute G*f(m), where f is called a mapping function. In general, this
    allows the inversion to optimize over a transformed or parameterized model
    space, rather than working directly with the mesh cell density values.

    Constructor inputs:

    mesh: discretize.TensorMesh -> mesh for the problem
    sensors: dict -> dictionary of sensors for the problem
    model_map: SimPEG.maps -> mapping from model to physical parameters.
    """
    def __init__(self,
        mesh: TensorMesh,
        sensors: Dict[Any, MuonSensor],
        model_map: maps.IdentityMap=maps.IdentityMap(), # For some reason, IdentityMap is the base class for all othe mappings
        **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.model_map = model_map
        self.mesh = mesh
        self.sensors = sensors

        # Store number of data for each sensor, for mapping
        # between vector of all data and dicts of data for each sensor.
        self.n_data = np.concatenate((
            [0],
            np.cumsum([len(sensor.rgrid_x)*len(sensor.rgrid_y)
                for sensor in sensors.values()])
        ))

        # Cell edges in each dimension
        self.xgrid = mesh.x0[0] + np.r_[0.0, np.cumsum(mesh.h[0])]
        self.ygrid = mesh.x0[1] + np.r_[0.0, np.cumsum(mesh.h[1])]
        self.zgrid = mesh.x0[2] + np.r_[0.0, np.cumsum(mesh.h[2])]

        # Get the forward operator
        self._G = self.get_G()

        nD = self._G.shape[0]
        self.survey = MuonSurvey(nD)

    def get_G(self):
        """
        Compute the forward operator G for the given
        mesh and sensors.

        Constructs G as a CSR sparse matrix directly.
        """
        nx, ny, nz = self.mesh.shape_cells
        nrows_G = sum([len(sensor.rgrid_x)*len(sensor.rgrid_y) for sensor in self.sensors.values()])
        nnz_est = int(nrows_G*np.sqrt(nx*ny*nz))
        nzvals = np.zeros(nnz_est)
        colinds = np.zeros(nnz_est, dtype=int)
        rowptrs = np.zeros(nrows_G+1, dtype=int)

        nnz = 0
        irow = 0
        for sensor in self.sensors.values():
            # Loop over the raypaths for this sensor
            ro = sensor.loc
            for ix in range(len(sensor.rgrid_x)):
                for iy in range(len(sensor.rgrid_y)):

                    # Get ray direction
                    rd = np.array([sensor.rgrid_x[ix], sensor.rgrid_y[iy], 1.0])
                    rd = rd/np.linalg.norm(rd)

                    # Get ray lengths through each cell in the full mesh
                    _, lvals = get_ray_intersection_pts(ro, rd, self.xgrid,
                        self.ygrid, self.zgrid)

                    # Build up G as a CSR sparse matrix directly
                    lvals = lvals.flatten(order='F') # Match SimPEG cell ordering
                    colinds_i = np.argwhere(lvals > 0.0).flatten()
                    nnz_i = len(colinds_i)
                    if nnz + nnz_i > nnz_est:
                        nnz_est = int(1.5*nnz_est)
                        nzvals = np.hstack((nzvals, np.zeros(nnz_est-nnz)))
                        colinds = np.hstack((colinds, np.zeros(nnz_est-nnz, dtype=int)))
                    nzvals[nnz:nnz+nnz_i] = lvals[colinds_i]
                    colinds[nnz:nnz+nnz_i] = colinds_i
                    nnz += nnz_i
                    rowptrs[irow+1] = nnz
                    irow += 1
        return sparse.csr_matrix((nzvals, colinds, rowptrs), shape=(nrows_G, nx*ny*nz))

    def get_J(self, m,f=None):
        """
        Return the sensitivity matrix.
        """
        return self._G @ self.model_map.deriv(m)

    # minimum functionalities for SimPEG simulation object
    def Jvec(self, m, v, f=None):
        """
        Return the sensitivity matrix multiplied by a vector.
        """
        return self._G @ self.model_map.deriv(m, v)


    def Jtvec(self, m, v, f=None):
        """
        Return the adjoint sens matrix multiplied by a vector.
        """
        return self.model_map.deriv(m).T @ (self._G.T @ v)

    def fields(self, m=None):
        """
        This method is required. For this problem, fields
        and data are the same.
        """
        return self._G @ self.model_map._transform(m)

    def get_data(self, m: np.ndarray) -> OrderedDict:
        """
        Compute the data for a given model m and split by sensor.
        This method is for convenience. SimPEG requires the data
        as a numpy array.
        """
        dall = self.fields(m)
        return OrderedDict([
            (key, dall[self.n_data[i]:self.n_data[i+1]])
                for i, key in enumerate(self.sensors.keys())
        ])

    def dpred(self, m, f=None):
        """
        Return the predicted data for a given model.
        """
        return self.fields(m)

    def residual(self, m, dobs, f=None):
        r"""
        The data residual:

        .. math::

            \mu_\\text{data} = \mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}

        :param numpy.ndarray m: geophysical model
        :param numpy.ndarray f: fields
        :rtype: numpy.ndarray
        :return: data residual
        """
        return self.dpred(m) - dobs

# SimPEG requires a survey class. More useful for problems
# where fields and data are different.
class MuonSurvey(BaseSurvey):
    def __init__(self, nD):
        self._nD = nD

    @property
    def nD(self):
        return self._nD
    @nD.setter
    def nD(self, value):
        self._nD = value

@numba.njit(error_model='numpy')
def get_ray_intersection_pts(ro: np.ndarray, rd: np.ndarray, xgrid: np.ndarray,
    ygrid: np.ndarray, zgrid: np.ndarray):
    """
    Given a ray and a grid, compute the intersection points of the ray with each
    cell in the grid and the length of the ray's path through each cell. Return
    numpy arrays holding the intersection points for each cell and the path lengths.
    """
    nx = len(xgrid) - 1; ny = len(ygrid) - 1; nz = len(zgrid) - 1
    intersections = np.full((nx, ny, nz, 2, 3), np.nan)
    lvals = np.zeros((nx, ny, nz))

    # Compute intersection of ray with domain boundaries
    # and from that which z slices of the mesh we have
    # to check for intersection with the ray.
    tminx, tmaxx = plane_col_intersections(ro, rd, 0, xgrid[0], xgrid[-1])
    tminy, tmaxy = plane_col_intersections(ro, rd, 1, ygrid[0], ygrid[-1])
    tminz, tmaxz = plane_col_intersections(ro, rd, 2, zgrid[0], zgrid[-1])
    tmax = min(tmaxx, tmaxy, tmaxz)
    z1 = ro[2]; z2 = ro[2] + tmax*rd[2]
    izmin = max(np.searchsorted(zgrid, np.minimum(z1, z2))-1, 0)
    izmax = min(np.searchsorted(zgrid, np.maximum(z1, z2)), nz-1)

    for iz in range(izmin, izmax+1): # Loop over columns
        # Get intersections with zmin and zmax planes
        tminz, tmaxz = plane_col_intersections(ro, rd, 2, zgrid[iz], zgrid[iz+1])
        tminz = max(tminz, 0.0)

        # Bound the search for cells that we need to check for intersection
        # with ray.
        x1 = ro[0] + tminz*rd[0]; x2 = ro[0] + tmaxz*rd[0]
        ixmin = max(np.searchsorted(xgrid, np.minimum(x1, x2))-1, 0)
        ixmax = min(np.searchsorted(xgrid, np.maximum(x1, x2)), nx-1)
        y1 = ro[1] + tminz*rd[1]; y2 = ro[1] + tmaxz*rd[1]
        iymin = max(np.searchsorted(ygrid, np.minimum(y1, y2))-1, 0)
        iymax = min(np.searchsorted(ygrid, np.maximum(y1, y2)), ny-1)

        # Loop over block of cells
        for ix in range(ixmin, ixmax+1):
            for iy in range(iymin, iymax+1):
                # Compute intersections with these cells
                x1 = xgrid[ix]; x2 = xgrid[ix+1]
                y1 = ygrid[iy]; y2 = ygrid[iy+1]
                tminx, tmaxx = plane_col_intersections(ro, rd, 0, x1, x2)
                tminy, tmaxy = plane_col_intersections(ro, rd, 1, y1, y2)
                tmin = max(tminx, tminy, tminz)
                tmax = min(tmaxx, tmaxy, tmaxz)
                if (tmax >= tmin) and (tmax >= 0.0):
                    # intersected_cols[ix, iy, iz] = True
                    tmin = tmin if tmin >= 0.0 else 0.0
                    intersections[ix, iy, iz, 0, :] = ro + tmin*rd
                    intersections[ix, iy, iz, 1, :] = ro + tmax*rd
                    lvals[ix, iy, iz] = np.abs(tmax - tmin)
    return intersections, lvals

@numba.njit(error_model='numpy')
def plane_col_intersections(ro, rd, idim, box_dmin, box_dmax):
    """
    Compute the intersection of a ray with axis-aligned planes.
    """
    # is ray parallel to planes?
    vd = rd[idim] # np.dot(rd, nrm_box)
    tmin = -np.inf; tmax = np.inf
    t0 = np.divide(box_dmin - ro[idim], vd)
    t1 = np.divide(box_dmax - ro[idim], vd)
    tmin = np.maximum(tmin, np.minimum(t0, t1))
    tmax = np.minimum(tmax, np.maximum(t0, t1))
    return tmin, tmax

