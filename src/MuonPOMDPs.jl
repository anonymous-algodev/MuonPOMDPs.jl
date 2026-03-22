module MuonPOMDPs

using Reexport
using Pkg
using NPZ
using HTTP
using JSON
using OrderedCollections
using LinearAlgebra
using Parameters
using Random
using StatsBase
using Distributions
using KernelDensity
using LocalFilters
using ImageFiltering
using NNlib
using BSON
using JLD2
@reexport using Statistics
@reexport using POMDPs 
@reexport using PyPlot
@reexport using PyCall

module Plots
	using Plots
    default(fontfamily="Computer Modern", framestyle=:box)
    import Plots: mm
end

export
    MuonParams,
    flatten_columns,
    download_data,
    load_data,
    boreholes!,
    plot_orebody_3d,
    plot_sections,
    plot_ground_truth,
    plot_state_action,
    get_mesh,
    forward,
    in_prism,
    get_rho_prism,
    plot_rho,
    plot_radiographs,
    inversion,
    get_state,
    plot_state,
    plot_ensembles,
    muon_data_keys,
    muon_data,
    muon_pca,
    pca,
    scree_plot,
    plot_pca,
    load_ensemble_matrix,
    download_ensemble_jsons,
    MuonState,
    MuonAction,
    MuonObservation,
    MuonObservations,
    MuonPOMDP,
    MuonBeliefUpdater,
    state_data,
    preprocess,
    plot_slice,
    plot_slices,
    nanmaximum,
    nanmean

global skfmm
global np
global stats

include("common.jl")
include("utils.jl")
include("intrusion.jl")
include("pomdp.jl")
include("pooling.jl")
include("belief.jl")
include("ensemble.jl")
include("pca.jl")
include("plotting.jl")

function __init__()
    if Sys.iswindows()
        ENV["PYTHON"] = "C:/PYTHON310/python.exe"
        Pkg.build("PyCall")
    end
    global skfmm = pyimport("skfmm")
    global np = pyimport("numpy")
    global stats = pyimport("scipy").stats
    if Sys.iswindows()
        py"""
import sys
sys.path.insert(0, $(joinpath(@__DIR__, "..")))
        """
    else
        pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, ".."))
    end
    py"""
from python.utils import *
from python.Tomo3D import *
from sklearn.decomposition import PCA
from discretize import TensorMesh
from SimPEG import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion
)
    """
    return nothing
end

end # module MuonPOMDPs
