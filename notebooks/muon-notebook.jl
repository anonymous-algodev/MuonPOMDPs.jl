### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 00150b0e-d3af-49d6-a7ed-f5ebcc7c47aa
begin
	using Pkg
	Pkg.develop(path="..")
end

# ╔═╡ 4fbf91b3-8fa0-41dc-8883-169bd247e76e
begin
	using Revise
	using MuonPOMDPs
end

# ╔═╡ e41b7c31-3bd3-467a-bf70-26a707168423
using PlutoUI; TableOfContents()

# ╔═╡ 9fa6b668-7e90-40d9-a41b-d0196ef975dc
using JSON

# ╔═╡ aa5859d2-1617-45fb-83ff-ac50eb972d93
using KernelDensity

# ╔═╡ cefe439a-857f-46b2-80b9-ffaf7c53ae37
using Clustering # kmeans

# ╔═╡ a69e711b-bea8-4f9e-8975-585a412728f3
module Plotting
	using Plots; default(fontfamily="Computer Modern", framestyle=:box)
end

# ╔═╡ 9ba12ded-ca90-48e1-a347-017885bb10c0
using Statistics

# ╔═╡ b47d19f6-b0b1-424f-85ec-6fb65ebe32c2
using Random

# ╔═╡ cb351292-c125-4000-86b9-06829d8e5d7c
using StatsBase

# ╔═╡ 7a8d663c-8db3-4a81-9c03-76b98986d685
using Distributions

# ╔═╡ 758be42b-d2d8-41ef-8ad4-a54150f9fe93
using NearestNeighbors

# ╔═╡ bb0e6eb9-9361-4f8e-b983-7b8c03b206e1
using LinearAlgebra

# ╔═╡ f4868da0-00c9-11ef-3146-0514f8342585
md"""
# Muon tomography POMDP
"""

# ╔═╡ bf34da5b-3c59-464e-b68b-7fdb2591d658
md"""
## Load data
"""

# ╔═╡ 501cba7c-b407-436c-80cb-b90bdfe99022
begin
	url_ground_truth = "https://www.dropbox.com/scl/fi/yt4d3xb3mj5v00trnhx3u/ground_truth_80x80x60.npy?rlkey=qo6hrni4sidp9nryy234g8zq5&dl=1"
	true_intrusion = load_data(url_ground_truth)
end; md"`true_intrusion`"

# ╔═╡ 332343df-6a00-4768-9443-63501889f1f1
begin
	url_topography = "https://www.dropbox.com/scl/fi/lua5fwx9dwejml14u3bxp/topography.npy?rlkey=2mrl452k7u0ubn0vwzfp61gm6&dl=1"
	topography = load_data(url_topography; multiplier=10)
end; md"`topography`"

# ╔═╡ 0458b701-6cb0-4f74-845b-1508861c06c6
md"""
## Data size parameters
"""

# ╔═╡ f987d4c8-47c8-4ea5-beac-6709ec4ed873
x_dim, y_dim, z_dim = n = size(true_intrusion)

# ╔═╡ 440bd7a1-3bb0-43ee-a251-bdd8ffa91f94
md"""
## 3d volume plots
"""

# ╔═╡ 47d418ac-4d06-41a4-9c67-9c469de5877e
md"""
## Topography
"""

# ╔═╡ 81ea665c-f80e-4337-864d-8b0f21031902
mp = MuonParams(true_intrusion, topography)

# ╔═╡ 95d47974-6b71-4d5e-97be-a168c73d7321
plot_orebody_3d(mp.intrusion; plt_title="Truth")

# ╔═╡ eb14aa82-ad11-4e85-9cb1-b863ba2ee46b
MuonPOMDPs.plot_sections(mp)

# ╔═╡ 33ba74e1-cf27-4cf2-90b1-0fbaa558f95e
plot_ground_truth(mp)

# ╔═╡ 9998a90a-3967-471f-901e-c59949ad1205
md"""
## Forward model muon tomography data

We add a simplified version of muon tomography to the test problem. The forward modelling is setup as follows. We define a collection of toy muon sensors. Each sensor has a location and collection of rays associated with it. Rays are straight paths originating from the sensor location and continuing out to infinity. The data for these sensors consist of the opacity (line integral of density) along each ray. In the true muon tomography problem we must relate opacity to observed muon counts. Here, for simplicity, we use the opacities as our observed data. This makes for a forward problem almost entirely equivalent to the toy seismic problem of straight ray traveltime tomography. In continuous space, the forward problem for a single ray is thus

$$\mathcal{O} = \int_L \rho \, \mathrm{d}\mathcal{l}$$

where $\mathcal{O}$ is opacity, $\rho$ is density, a scalar function of position, and $L$ is the straight path from the sensor location to the point where the ray intersects the domain boundary. In a real application we must insure the domain is large enough that rays always intersect with topography and not the side of the domain. For this synthetic example, we ignore this restriction, in order to reduce the size of the domain. This is ok as long as our simulated true opacity data are generated on the same domain.

We discretize the problem onto a SimPEG tensor product mesh. Densities are assumed constant in each mesh cell. This gives a linear forward problem

$$\mathbf{d} = \mathbf{G}\boldsymbol{\rho}$$

where $\mathbf{d}$ is a vector consisting of all the opacities for all the sensors, $\boldsymbol{\rho}$ is vector of cell densities, and $\mathbf{G}$ is an extremely sparse matrix with non-zero values representing the lengths travelled by each ray through the mesh cells.
"""

# ╔═╡ 7db19600-2add-4d99-9747-73e1a933e7ac
muon_sim_simpeg, is_active = forward(mp)

# ╔═╡ f7ff14b7-f62f-4a68-a65c-c7749fc76e1a
mesh = get_mesh(mp)

# ╔═╡ e50b4d8a-30ff-4653-a5b3-0c7c65e0b7d5
# Null model (no intrusion) with density of 1 for demonstration purposes
rho0 = ones(mesh.nC)

# ╔═╡ 43ea9f22-e78d-43fa-8834-e40fe3775784
# # Forward model the null model
d0_muon = muon_sim_simpeg.get_data(rho0[is_active])

# ╔═╡ 52ab2575-ce03-4713-9d5e-850cfea18624
begin
	# True model, with intrusion density = 1.5, background density 1 and air density 0
	rho_true = deepcopy(mp.intrusion)
	rho_true[rho_true .== 1] .= 1.5
	rho_true[rho_true .== 0] .= 1.0
end

# ╔═╡ 004c2c60-8a91-42a3-abfd-d628908c2166
begin
	# Forward model the true data and assign uncertainties
	# For this toy fwd problem, for now, the uncertainties are set to 0.5% of the true data
	# values. In the true problem, uncertainties are based on the statistics of muon
	# flux and on the detector properties. How to most usefully set them for this
	# toy problem is an open question.
	dtrue_muon = muon_sim_simpeg.get_data(flatten_columns(rho_true)[is_active])
	std_muon = MuonPOMDPs.OrderedDict([(k, 0.005*v) for (k,v) in dtrue_muon])
end

# ╔═╡ 9799f5c4-896a-44a2-94f4-31e82a9bf8e3
dtrue_muon

# ╔═╡ e469dfff-a6f6-44ad-a7c5-911abca8aa23
rho_prism = get_rho_prism(mp, mesh);

# ╔═╡ ef4c2df0-c967-472d-aa8c-837a6670fb0e
# Forward model the prism model
dprism_muon = muon_sim_simpeg.get_data(flatten_columns(rho_prism)[is_active]);

# ╔═╡ c337b86e-bfb5-42c5-9e02-f39be62f5f4b
plot_rho(mp, rho_prism)

# ╔═╡ 69d3331d-6ce6-4a13-a792-d31bfa94242a
plot_radiographs(d0_muon)

# ╔═╡ 7481e01c-2edf-4fb8-8420-4c6544e17729
plot_radiographs(dprism_muon)

# ╔═╡ cd9d4ac3-fc9d-4f69-ad30-0308cf1c3054
# Plot (dtrue-d0)/std radiographs:
plot_radiographs(dprism_muon, id->(dtrue_muon[id] .- d0_muon[id]) ./ std_muon[id])

# ╔═╡ 978287f6-89ae-409f-ab48-a401ac7cc539
# Plot average density radiographs (i.e. opacity divided by pathlength)
# Since density is 1 in the null model, d0 is just the pathlength.
plot_radiographs(dprism_muon, id->dtrue_muon[id] ./ d0_muon[id])

# ╔═╡ 60aaa38a-b53d-43f9-bf6b-549e75c69065
md"""
# True muon data
"""

# ╔═╡ c1d987ed-173d-4950-ac57-4db76cf1adf3
sort(dtrue_muon)

# ╔═╡ 489dc354-fe33-4b71-931f-f85fbafb9327
md"""
## SimPeg Deterministic Muon Tomography Inversion
"""

# ╔═╡ 71e4d629-779c-41c7-89bb-e24f3bb261f1
# recovered_model = inversion(mp, dtrue_muon, std_muon, muon_sim_simpeg, rho0, is_active, mesh)

# ╔═╡ 55c89c82-f488-45e4-b60e-88178212f4b7
md"""
# Ensembles and PCA
"""

# ╔═╡ f2f60067-9615-405b-8c0f-24f38cd7ee6d
begin
	m_ensemble = load_ensemble_matrix()
end; md"`m_ensemble`"

# ╔═╡ fe789494-2808-41c9-8a0c-46bf511f4777
size(m_ensemble)

# ╔═╡ 241c7878-9ffc-402a-93ab-ab96072f3eae
md"""
`m_ensemble`: the ensemble of prior state models. ,
- It has totally 400 realizations, each model has 384000 (80x80x60) dimesnions ,
- each model dimension is 80(`X_dim`) x 80(`Y_dim`) x 60(`Z_dim`),
- 1: is the intrusion model; 0: is the host rock
"""

# ╔═╡ 53dc39aa-59d4-4be4-bca9-3d3621b1209b
# total number of prior model realizations
n_reals = size(m_ensemble, 1)

# ╔═╡ abc47edd-c248-4cc7-a4b4-52f8d73a4e0d
md"""
Plot the samples of the prior model ensemble.
"""

# ╔═╡ 22699d45-766c-40b3-9bfe-0dd9cf995ee9
plot_ensembles(mp, m_ensemble)

# ╔═╡ 17c4165e-6635-48b7-b855-c5cb6b57b9dc
get_state(mp, m_ensemble, 1) |> size

# ╔═╡ 51826f7a-22d8-4ee4-8bfb-66305ce632d5
plot_state(mp, m_ensemble, 10)

# ╔═╡ 29d4ce98-9350-47dc-afb9-1ca4d0f15c41
# begin
# 	download_data("https://www.dropbox.com/scl/fo/u51s3i7xthbrm36lkxtyc/AJ0gdeOSS6lvOi4txPOqecU/muon?rlkey=xv04caprqeej5dnup9of1ec3v&subfolder_nav_tracking=1&st=u5yvxc3z&dl=1", "muon.zip")
# 	run(`unzip muon.zip -d muon`)
# end

# ╔═╡ 638f89a6-9f3b-4fc7-a702-63cb96b4a002
Dict(i=>i for i in 1:10)

# ╔═╡ ec91ff9d-0858-4c9b-b86b-146157f1560e
md"""
## Belief updating (inversion)
Also called _Bayesian updating_.
"""

# ╔═╡ ad5988be-b0a6-4c57-bfdb-9981c0e58a81
sensor_num = 6;

# ╔═╡ 5a919615-82ef-4981-bbf5-514273e6d463
begin
	d_ensemble = muon_data(n_reals; sensor_num)
	n_keys = length(muon_data_keys())
end;

# ╔═╡ c876ed50-8285-4a04-9b1a-30e7433a0b76
d_ensemble

# ╔═╡ 5abac9ba-fdf1-4554-80f1-1e505665fa00
md"""
## Multivariate KDE
"""

# ╔═╡ 32386a2c-2573-4507-87a6-99d131f1a1d8
stats = MuonPOMDPs.pyimport("scipy").stats;

# ╔═╡ 84a52967-6c8c-45b3-b54b-3760bcc4a7df
md"""
## PCA subplots
"""

# ╔═╡ 7d991555-5269-47bb-a608-4fd898cda285
md"""
## PCA clustering
"""

# ╔═╡ b2c89833-2840-4471-b023-2fa22f8b20bf
@bind k Slider(2:10, default=4, show_value=true)

# ╔═╡ ba7adbb4-f816-4f48-8705-458213163e3c
# ╠═╡ disabled = true
#=╠═╡
plot_pca_k(d_pca, d_pcscr, k; run_kmeans=false, run_kde=true)
  ╠═╡ =#

# ╔═╡ 38217246-b987-4693-9a25-c5b88bbb957b
md"""
# Plotting module
"""

# ╔═╡ 6568d8cf-d572-4cea-9929-0a387684f1c8
Plots = (@__MODULE__).Plotting

# ╔═╡ df911be9-bfb9-4e39-bdb7-fb3e2f525e9a
function plot_pca_single(d_pcscr, i, j; show_contour=false, univariate=false)
	kwargs = (c=:white,
              widen=false,
              ms=3,
              msw=1,
              size=(400,380),
              label=false)
	if show_contour
		dX = d_pcscr[:,i]
		dY = d_pcscr[:,j]
		if univariate
			U1 = kde(dX)
			U2 = kde(dY)
			f = (x,y)->pdf(U1, x)*pdf(U2,y)
		else
			U = stats.gaussian_kde(hcat(d_pcscr[:,i], d_pcscr[:,j])')
			f = (x,y)->U([x,y])[1]
		end
		xl = extrema(dX)
		yl = extrema(dY)
		X = range(xl..., length=100)
		Y = range(yl..., length=100)
		Plots.contourf(X, Y, f, c=:viridis, colorbar=false)
		return Plots.scatter!(d_pcscr[:,i], d_pcscr[:,j]; kwargs...)
	else
		return Plots.scatter(d_pcscr[:,i], d_pcscr[:,j]; kwargs...)
	end
end

# ╔═╡ a792e62e-a1a0-4177-bfcd-0c7e5e6b7197
function subplot_pca(d_pcscr, i, j; show_contour=false)
	pca_fig = plot_pca_single(d_pcscr, i, j; show_contour)
	xl = Plots.xlims()
	yl = Plots.ylims()
	YL = range(yl[1], yl[2], length=100)
	lay = Plots.@layout [a{0.3h} _ _; b{0.7h, 0.45w} c{0.22w} _]

	U1 = kde(d_pcscr[:,i])
	U2 = kde(d_pcscr[:,j])

	top_fig = Plots.plot(x->pdf(U1, x), 
		xlims=xl,
		label="PC1",
		c=:darkred,
		x_foreground_color_text=false)
	side_fig = Plots.plot([pdf(U2, x) for x in YL], YL,
		ylims=yl,
		label="PC2",
		c=:darkred,
		y_foreground_color_text=false,
		rotation=-90)
	return Plots.plot(top_fig, pca_fig, side_fig, layout=lay)
end

# ╔═╡ 14b3bf7d-bc78-4c21-a75e-eae7c3465e9f
function plot_pca_k(d_pca, d_pcscr, k=4;
		colors=[:red, :lightblue], run_kmeans=false, run_kde=false)
	group = []
	explained_var = d_pca.explained_variance_ratio_
	for i in 1:k
		for j in 1:k
			local pi = round(100*explained_var[j], sigdigits=3)
			yl = (j == 1 && k ≤ 4) ? "PC$i ($pi%)" : ""
			local pj = round(100*explained_var[j], sigdigits=3)
			xl = (i == k && k ≤ 4) ? "PC$j ($pj%)" : ""
			if i == j
				local plt = Plots.plot(
					x_foreground_color_border=:lightgray, y_foreground_color_border=:lightgray,
					x_foreground_color_text=false,
					y_foreground_color_text=false,
					x_foreground_color_axis=false,
					y_foreground_color_axis=false,
					xlims=extrema(d_pcscr[:,j]),
					ylims=extrema(d_pcscr[:,i]),
					grid=false,
					# ticks=false,
					xlabel=xl, ylabel=yl)
				push!(group, plt)
			else
				if run_kmeans
					R = kmeans(d_pcscr[:,[j,i]]', 2)
					c = map(a->colors[a], assignments(R))
				else
					c = :white
				end
				if run_kde
					U = stats.gaussian_kde(hcat(d_pcscr[:,j], d_pcscr[:,i])')
					f = (x,y)->U([x,y])[1]
					dX = d_pcscr[:,j]
					dY = d_pcscr[:,i]
					xrange = extrema(dX)
					yrange = extrema(dY)
					X = range(xrange..., length=100)
					Y = range(yrange..., length=100)
					Plots.contourf(X, Y, f, c=:viridis, colorbar=false)
					scatter_plot = Plots.scatter!
					widen = false
				else
					scatter_plot = Plots.scatter
					widen = true
				end
				scatter_plot(d_pcscr[:,j], d_pcscr[:,i]; c=c, ms=2, msw=0.3, widen)
				push!(group,
					Plots.plot!(size=(300,300),legend=false,xlabel=xl,ylabel=yl))
			end
		end
	end
	return Plots.plot(group..., ticks=k ≤ 4, layout=(k,k), size=(1000,950))
end

# ╔═╡ f58c2ce0-8707-4899-8440-0f40eb9768ba
md"""
# POMDP
"""

# ╔═╡ d6669a2b-c5f4-48d0-9127-721751d400f9
pomdp = MuonPOMDP(mp)

# ╔═╡ eb3a3d9c-94ee-472d-b0fa-49e73868086a
pomdp.prior

# ╔═╡ adf2d25c-95b5-48a9-8d3c-f7268e5ae724
ds0 = initialstate(pomdp); # initial state distribution

# ╔═╡ fbd4846c-3ab7-42a7-982b-91e4b66bd99b
s0 = rand(ds0); # initial state

# ╔═╡ 4afa5161-28a6-4528-a50f-90fc02960cd6
md"""
## Initial belief $b_0 = \{s \mid s \sim \mathcal{S}_0\}$
"""

# ╔═╡ 1a442c43-d1a1-485e-b00a-bcf689a323a4
md"""
## Sample state realization $s \sim \mathcal{S}_0$
"""

# ╔═╡ 2381e101-b291-4d35-9102-776633a4ddc9
plot_state(state_data(s0))

# ╔═╡ d087317a-9f95-4748-85cd-fa224d964e01
md"""
## Belief mean and std. 3D plots
"""

# ╔═╡ b6c50e8d-b7a1-4168-a59b-10e4cee23a87
plot_state(state_data(s0); topdown=true)

# ╔═╡ 4e288174-c167-49ea-8539-1431e1d6964d
md"""
## Slice plots
"""

# ╔═╡ 0e06448f-753d-4ed9-8b17-5454bd3492d3
nanmaximum(X) = maximum(x->isnan(x) ? -Inf : x, X)

# ╔═╡ fe9f265d-d65b-4ada-908e-a563f623d896
nanmean(X) = mean(x->isnan(x) ? 0 : x, X)

# ╔═╡ 1660b021-21de-4501-b486-562482bd2922
plot_slice(s::MuonState; kwargs...) = plot_slice(state_data(s); kwargs...)

# ╔═╡ 3d275a41-e6a5-48bb-9818-20692615499d
function plot_slice(data; mode=nanmaximum, dims=3, from_slices=false)
	Z = mapslices(mode, data; dims)
	Z = dropdims(Z, dims = (findall(size(Z) .== 1)...,))
	plt_xy = Plots.heatmap(Z', c=:viridis, ratio=1)
	if length(unique(size(Z))) == 1 # only for m×m data
		# if from_slices
			# Plots.ylims!(Plots.xlims())
		# else
			Plots.xlims!(Plots.ylims())
		# end
	else
		lims = max(Plots.xlims()[2], Plots.ylims()[2])
		Plots.xlims!(1, lims)
		Plots.ylims!(1, lims)
	end
	xlab = Dict(1=>"y", 2=>"x", 3=>"x")
	ylab = Dict(1=>"z", 2=>"z", 3=>"y")
	return Plots.plot!(xlabel=xlab[dims], ylabel=ylab[dims])
end

# ╔═╡ 0f102029-3e59-4f40-a938-0cf9c8c39814
function plot_slices(data; mode=nanmaximum, title="")
	plts = []
	for dims in [3, 1, 2]
		push!(plts, plot_slice(data; mode, dims, from_slices=isempty(title)))
	end
	return Plots.plot(plts...;
		layout=(1,3), size=(600,200), colorbar=false, plot_title=title,
		topmargin=5Plots.Plots.mm, bottommargin=5Plots.Plots.mm,
		plot_titlefontsize=11)
end

# ╔═╡ 91f0626e-470e-4b4b-87e9-45a92b77dcd1
md"""
### Binary (true) intrusion
"""

# ╔═╡ cad3c198-bd9e-4da2-aeee-66d2fe25f795
plot_slices(mp.intrusion; title="True intrusion"); Plots.savefig("plot_true_intrusion.png"); Plots.plot!()

# ╔═╡ b416a8db-dae0-4dcd-9197-4ee517c9d793
md"""
### Mean intrusion
"""

# ╔═╡ 7cb05d98-764a-4391-a89f-6a855163f6f0
plot_slices(mp.intrusion; mode=nanmean)

# ╔═╡ 49e8fc18-ccdc-4ebe-8cb7-30379b25a969
md"""
### Mean single state
"""

# ╔═╡ 12be5e6a-71bc-4980-82ac-2f011b5a20ea
plot_slices(s0; mode=nanmean)

# ╔═╡ e646d365-3899-4dd6-b4cf-8fad6bac2acf
md"""
### Belief mean
"""

# ╔═╡ 6d8b731a-f177-4b51-91d4-ff65bc22c8fd
md"""
### Belief std.
"""

# ╔═╡ 2b4084e5-f059-4029-b524-080a6eb5dd28
md"""
## `POMDPs.gen`
"""

# ╔═╡ 186ca4a4-098d-4831-89cc-8400afa6c9dc
ap = rand(actions(pomdp))

# ╔═╡ 72c02cb5-0bdf-4ea2-99fd-abbc2be3c27f
sp, o, r = @gen(:sp, :o, :r)(pomdp, s0, ap)

# ╔═╡ 73fad540-f513-4304-9e33-040fc233ce38
o[end].sensor_num

# ╔═╡ af5df3bc-c292-45e6-b340-6efce2c296d5
md"""
## Observation weight $O(o \mid a, s')$
"""

# ╔═╡ 172e4c70-ab43-47ef-9721-e33219400eba
md"""
## Perturb
"""

# ╔═╡ dfd72215-d461-4f0d-beaa-6a362b762139
# kdtree = KDTree(s; leafsize=10)

# ╔═╡ 1330b912-214f-4ef6-b794-a8be6465fedc
s = state_data(s0);

# ╔═╡ c50984d1-deaf-49b2-804e-b0ba01716fa7
function perturb(s)
	return s .* rand(0:1, size(s)...) # random {0,1} noise
	# sp = ds0[argmin(map(s0->s == s0.data ? Inf : norm(s - s0.data, 2), ds0))]
	# return sp.data
end

# ╔═╡ a268b5df-fb29-492a-bf44-922774d3a5fe
function perturb(b::Vector{MuonState})
	return map(s->MuonState(perturb(s.data), s.index), b)
end

# ╔═╡ 2c178473-6618-4b50-9284-bfcdf448c1d2
argmin(map(s0->s == s0.data ? Inf : norm(s - s0.data, 2), ds0))

# ╔═╡ 8901f8c4-420a-4ed2-9740-6c82d9f86cf4
plot_slice(perturb(s)); Plots.plot!(size=(600*0.75,400*0.75))

# ╔═╡ 1c84c6e8-d34c-474b-8275-266ddd5c3fc3
md"""
## Updated belief (one sensor) $b'$
Update initial belief $b_0$ with a single action $a$ (i.e., one sensor) to get $b'$.
"""

# ╔═╡ 9dd21e83-b280-4589-9041-980382538113
md"""
### Updated belief mean
"""

# ╔═╡ 5ff4d10a-f683-4841-8f1a-cb1248d9f660
md"""
### Updated belief std.
"""

# ╔═╡ b9058e4b-9a98-4847-9240-4b2b2ed3d410
md"""
## Belief updater
"""

# ╔═╡ e68912e1-68db-41ba-9543-9c44b199b2ff
up = MuonBeliefUpdater(pomdp, 400, 5);

# ╔═╡ 80d2da5f-ca1c-45ff-95f3-102c75ca6b47
begin
	Random.seed!(0)
	b0 = initialize_belief(up, ds0)
end;

# ╔═╡ 23fa80bd-b3c0-477b-9afb-5098b9415a0a
bk = map(s->s.index, b0);

# ╔═╡ 72e149e4-5113-41fb-8cec-49fe01e72a03
# d_pca, d_pcscr = pca(d_ensemble; n_components=100)
d_pca, d_pcscr = pca(d_ensemble[bk]; n_components=100)

# ╔═╡ 056aeceb-123c-4c4d-8d8c-088f68af8ea6
scree_plot(d_pca)

# ╔═╡ 91f05e4a-eb40-4d16-af53-58aba1082adb
sum(d_pca.explained_variance_ratio_[1:2]) # explained variance of PC1 and PC2

# ╔═╡ 8674af05-02b3-4663-85bb-eb466eca5772
plot_pca(d_pca, d_pcscr)

# ╔═╡ 4fb105e7-c3b6-43bb-9606-f8fde1a4fb2c
begin
	K = collect(keys(sort(dtrue_muon)))
	dtrue_data = dtrue_muon[K[sensor_num]]
	D = reshape(dtrue_data, 1, :) # (n_samples, n_features)
	Dp = d_pca.transform(D)
end

# ╔═╡ f8fb8b77-7ed5-4aa8-a46f-b5c414b31266
begin
	pc1 = 1
	pc2 = 2
	plot_pca_single(d_pcscr, pc1, pc2)
	Plots.scatter!(Dp[:,pc1], Dp[:,pc2], c="#FEC51D", mark=:square, label="truth")
end

# ╔═╡ 624138e2-7885-43f6-ac3c-47cc5e3ec6e3
U = stats.gaussian_kde(hcat(d_pcscr[:,pc1], d_pcscr[:,pc2])');

# ╔═╡ 8c33bcc9-b1bc-46a8-9f0d-9b9739a396e9
subplot_pca(d_pcscr, 1, 2; show_contour=true)

# ╔═╡ 2f4c9979-ee02-44e7-963a-68fd8a4ab1f6
plot_pca_single(d_pcscr, 1, 3; show_contour=true)

# ╔═╡ aadb0624-b33a-4ab2-a648-dd388d2fa43f
Markdown.parse("\$\\text{explained variance} = $(100* round(sum(d_pca.explained_variance_ratio_[1:k]), digits=3)) \\%\$")  # explained variance of PCs 1-k

# ╔═╡ 21b9b85e-ca6a-4f78-9272-e4dc5fa0fe36
plot_pca_k(d_pca, d_pcscr, k; run_kmeans=true, run_kde=false)

# ╔═╡ b3c1fdf5-a298-4398-8a26-61f81ef8404e
p = MuonPOMDPs.observation_weight(dtrue_muon, d_pca, U, [1, 2], sensor_num)

# ╔═╡ 0aabb462-fc48-49da-88a6-6305ffd1747c
W′ = MuonPOMDPs.reweight(d_pcscr, U, [pc1, pc2], p);

# ╔═╡ 844bb0d4-0cd6-48c0-b948-a61114e6263f
begin
	Plots.histogram(W′, size=(600,300), c=:lightgray, label="particle weight")
	Plots.ylims!(0, Plots.ylims()[2])
end

# ╔═╡ 8729d29e-8d14-4278-82d6-9ec90f2f07d4
plot_state(mean(state_data(b0))); MuonPOMDPs.title("belief mean"); MuonPOMDPs.gcf()

# ╔═╡ be3c6b62-7657-4b59-adcb-eacd1ba1d094
plot_state(std(state_data(b0))); MuonPOMDPs.title("belief std"); MuonPOMDPs.gcf()

# ╔═╡ 173768ba-2f79-4baf-a8aa-0846c67b7793
plot_slices(mean(state_data(b0)); mode=nanmean); Plots.savefig("initial_belief_mean_slides.png"); Plots.plot!()

# ╔═╡ 4e3e6e46-c396-47fd-a16b-44a566dcf608
plot_slices(std(state_data(b0)); mode=nanmean); Plots.savefig("initial_belief_std_slides.png"); Plots.plot!()

# ╔═╡ b9e3ea9e-e62d-4d26-94b4-eef458db9c93
bp = sample(b0, StatsBase.Weights(W′), length(b0), replace=true);

# ╔═╡ ef3bbba2-694c-4054-b023-fb541f1e3159
plot_slices(mean(state_data(bp)); mode=nanmean)

# ╔═╡ 9fc31e25-4cec-4ad8-83a5-9db9a86efbf2
plot_slices(std(state_data(bp)); mode=nanmean)

# ╔═╡ 0cc43cc3-2e36-442d-acaa-ed0e628ab82c
md"""
## Iterative belief updating $b_{1:T}$
"""

# ╔═╡ d38485e0-bf7b-4b82-b1a3-4edd13dd35cc
function sim_belief_update(b0; max_steps=5, n_pcs=2)
	b = b0
	B = [b]
	pcs = 1:n_pcs
	time = @elapsed for t in 1:max_steps
		sensor_num = t
		bk = map(s->s.index, b)
		# @info t, length(unique(bk))
		d_pca, d_pcscr = pca(d_ensemble[bk]; n_components=n_pcs)
		values = hcat([d_pcscr[:,pc] for pc in pcs]...)'
		try
			U = stats.gaussian_kde(values)
			p = MuonPOMDPs.observation_weight(dtrue_muon, d_pca, U, pcs, sensor_num)
			W′ = MuonPOMDPs.reweight(d_pcscr, U, pcs, p)
			b = sample(b, StatsBase.Weights(W′), length(b), replace=true)
			push!(B, b)
		catch err
			@info "Belief updating error (sensor_num = $sensor_num)."
		end
	end
	@info "Average belief update: $(time/max_steps) seconds"
	return B
end

# ╔═╡ 00e84e78-07b9-4a5d-9a03-219b4e06e52b
B = sim_belief_update(b0; max_steps=45, n_pcs=5);

# ╔═╡ a13efe5f-fe95-4d28-84be-63cadf64e356
md"""
### Final belief mean
"""

# ╔═╡ 2884b072-cddc-4441-938a-3e4953814b26
plot_slices(mean(state_data(B[end])); mode=nanmean, title="Belief mean"); Plots.savefig("plot_final_belief_mean.png"); Plots.plot!()

# ╔═╡ 0a4032eb-d8a1-444c-91d0-ac553cc0dc99
md"""
### Final belief std.
"""

# ╔═╡ e8044aad-4832-4ff4-be1b-c7892acafda0
plot_slices(std(state_data(B[end])); mode=nanmean, title="Belief standard deviation"); Plots.savefig("plot_final_belief_std.png"); Plots.plot!()

# ╔═╡ 78ab85a9-4b07-4464-9749-6a0de5f45863
md"""
### True intrusion mean
"""

# ╔═╡ b7429454-b090-4014-b713-d3434f74059c
plot_slices(mp.intrusion; mode=nanmean) # truth

# ╔═╡ 04eb47bd-1888-43ac-8955-1db2a597b5fc
plot_slices(abs.(mean(state_data(B[end])) - mp.intrusion); mode=nanmean, title="\$|b - s_\\textrm{true}|\$")

# ╔═╡ 203a21ee-dd8e-4eaf-bae3-3d17fb4bdddc
md"""
> **TODO**: `mean(b::Vector{MuonState})` and `std(b::Vector{MuonState})`
"""

# ╔═╡ ab42779d-5adf-4a64-a4b4-b4f2ae25ec3c
md"""
## Action (drilling + sensing) plots
"""

# ╔═╡ 430a683b-b457-45c1-b18d-49f240021a34
md"""
> Animate camera, try Plotly
"""

# ╔═╡ b1f23586-a8b7-4cec-9413-bac8c4cbabc5


# ╔═╡ 7760b280-ee74-4453-ae92-bd9a7b0be230
plot_state_action(mp, state_data(s0))

# ╔═╡ 4f515f9c-bed7-4b15-946d-551712efacff
pomdp.prior

# ╔═╡ a99b80b2-589b-441d-81d6-40cc8a910364
function sensornum2key(pomdp::MuonPOMDP, sensor_num)
	return pomdp.sensor_keys[sensor_num]
end

# ╔═╡ cd142fca-e547-4814-9c93-4c79097d3c6b
function sensorindex2keys(pomdp, sensor_idx)
	K = pomdp.sensor_keys
	I = findall(k->occursin("$(sensor_idx-1)_", k), K)
	return K[I]
end

# ╔═╡ b9c571f3-89bc-4590-862a-3abd7b04d8b0
function sensorkeys2nums(pomdp::MuonPOMDP, keys)
	return map(k->findfirst(pomdp.sensor_keys .== k), keys)
end

# ╔═╡ f5adb716-bf25-42f3-a0c5-fa5fb2fecf2c
kk = sensorindex2keys(pomdp, 3)

# ╔═╡ 1923f292-08de-4b05-8404-dd0a8da603f2
nn = sensorkeys2nums(pomdp, kk)

# ╔═╡ a50c7e75-5fd2-4634-9e24-d1855266c8a6
pomdp.sensor_keys[nn]

# ╔═╡ 0535599d-9fa5-4890-8f7a-46ff6fb9dce7
a = rand(actions(pomdp))

# ╔═╡ 00dc7252-f661-491c-af28-45edc1e861fc
b′ = update(up, b0, a, o);

# ╔═╡ 010d94d0-e6c3-456a-8303-66f5f5997048
plot_slices(mean(state_data(b′)); mode=nanmean, title="Belief mean (updater)")

# ╔═╡ 2634bdaa-495f-456d-a4a5-727e0da5c3cb
sensornum2key(pomdp, sensor_num)

# ╔═╡ 10d5560d-30f1-4e4e-8773-f79d977db866
begin
	a0 = rand(actions(pomdp))
	@show [a0.x, a0.y, a0.i]
	boreholes!(mp, [a0.x], [a0.y])
	plot_ground_truth(mp)
	MuonPOMDPs.savefig("truth_action.png")
	ax2 = plot_state_action(mp, mean(state_data(b0)))
	MuonPOMDPs.savefig("belief_action.png")
	md"""
	Truth | Belief Mean
	:---------------:|:----:
	$(LocalResource("truth_action.png")) | $(LocalResource("belief_action.png"))
	"""
end

# ╔═╡ 8b505562-6c75-4476-9767-2afca19f2e42
a0

# ╔═╡ 31a4dffe-46bf-4e06-8b44-c625e6ee5b56
begin
	# reset
	mp.xbh = [200, 1400, 2100, 2600, 3800]
	mp.ybh = fill(2500, length(mp.xbh))
	mp.lengths = [950.0, 1050.0, 1100.0, 1100.0, 1150.0]
	boreholes!(mp)
	plot_ground_truth(mp)
end

# ╔═╡ Cell order:
# ╟─f4868da0-00c9-11ef-3146-0514f8342585
# ╠═00150b0e-d3af-49d6-a7ed-f5ebcc7c47aa
# ╠═4fbf91b3-8fa0-41dc-8883-169bd247e76e
# ╠═e41b7c31-3bd3-467a-bf70-26a707168423
# ╟─bf34da5b-3c59-464e-b68b-7fdb2591d658
# ╟─501cba7c-b407-436c-80cb-b90bdfe99022
# ╟─332343df-6a00-4768-9443-63501889f1f1
# ╟─0458b701-6cb0-4f74-845b-1508861c06c6
# ╠═f987d4c8-47c8-4ea5-beac-6709ec4ed873
# ╟─440bd7a1-3bb0-43ee-a251-bdd8ffa91f94
# ╠═95d47974-6b71-4d5e-97be-a168c73d7321
# ╟─47d418ac-4d06-41a4-9c67-9c469de5877e
# ╠═81ea665c-f80e-4337-864d-8b0f21031902
# ╠═eb14aa82-ad11-4e85-9cb1-b863ba2ee46b
# ╠═33ba74e1-cf27-4cf2-90b1-0fbaa558f95e
# ╟─9998a90a-3967-471f-901e-c59949ad1205
# ╠═7db19600-2add-4d99-9747-73e1a933e7ac
# ╠═f7ff14b7-f62f-4a68-a65c-c7749fc76e1a
# ╠═e50b4d8a-30ff-4653-a5b3-0c7c65e0b7d5
# ╠═43ea9f22-e78d-43fa-8834-e40fe3775784
# ╠═52ab2575-ce03-4713-9d5e-850cfea18624
# ╠═004c2c60-8a91-42a3-abfd-d628908c2166
# ╠═9799f5c4-896a-44a2-94f4-31e82a9bf8e3
# ╠═e469dfff-a6f6-44ad-a7c5-911abca8aa23
# ╠═ef4c2df0-c967-472d-aa8c-837a6670fb0e
# ╠═c337b86e-bfb5-42c5-9e02-f39be62f5f4b
# ╠═69d3331d-6ce6-4a13-a792-d31bfa94242a
# ╠═7481e01c-2edf-4fb8-8420-4c6544e17729
# ╠═cd9d4ac3-fc9d-4f69-ad30-0308cf1c3054
# ╠═978287f6-89ae-409f-ab48-a401ac7cc539
# ╟─60aaa38a-b53d-43f9-bf6b-549e75c69065
# ╠═c1d987ed-173d-4950-ac57-4db76cf1adf3
# ╟─489dc354-fe33-4b71-931f-f85fbafb9327
# ╠═71e4d629-779c-41c7-89bb-e24f3bb261f1
# ╟─55c89c82-f488-45e4-b60e-88178212f4b7
# ╠═f2f60067-9615-405b-8c0f-24f38cd7ee6d
# ╠═fe789494-2808-41c9-8a0c-46bf511f4777
# ╟─241c7878-9ffc-402a-93ab-ab96072f3eae
# ╠═53dc39aa-59d4-4be4-bca9-3d3621b1209b
# ╟─abc47edd-c248-4cc7-a4b4-52f8d73a4e0d
# ╠═22699d45-766c-40b3-9bfe-0dd9cf995ee9
# ╠═17c4165e-6635-48b7-b855-c5cb6b57b9dc
# ╠═51826f7a-22d8-4ee4-8bfb-66305ce632d5
# ╠═29d4ce98-9350-47dc-afb9-1ca4d0f15c41
# ╠═5a919615-82ef-4981-bbf5-514273e6d463
# ╠═9fa6b668-7e90-40d9-a41b-d0196ef975dc
# ╠═638f89a6-9f3b-4fc7-a702-63cb96b4a002
# ╠═72e149e4-5113-41fb-8cec-49fe01e72a03
# ╠═c876ed50-8285-4a04-9b1a-30e7433a0b76
# ╠═eb3a3d9c-94ee-472d-b0fa-49e73868086a
# ╠═056aeceb-123c-4c4d-8d8c-088f68af8ea6
# ╠═91f05e4a-eb40-4d16-af53-58aba1082adb
# ╠═8674af05-02b3-4663-85bb-eb466eca5772
# ╟─ec91ff9d-0858-4c9b-b86b-146157f1560e
# ╠═aa5859d2-1617-45fb-83ff-ac50eb972d93
# ╠═ad5988be-b0a6-4c57-bfdb-9981c0e58a81
# ╠═4fb105e7-c3b6-43bb-9606-f8fde1a4fb2c
# ╠═f8fb8b77-7ed5-4aa8-a46f-b5c414b31266
# ╟─5abac9ba-fdf1-4554-80f1-1e505665fa00
# ╠═32386a2c-2573-4507-87a6-99d131f1a1d8
# ╠═624138e2-7885-43f6-ac3c-47cc5e3ec6e3
# ╟─84a52967-6c8c-45b3-b54b-3760bcc4a7df
# ╠═8c33bcc9-b1bc-46a8-9f0d-9b9739a396e9
# ╠═2f4c9979-ee02-44e7-963a-68fd8a4ab1f6
# ╠═df911be9-bfb9-4e39-bdb7-fb3e2f525e9a
# ╟─a792e62e-a1a0-4177-bfcd-0c7e5e6b7197
# ╟─7d991555-5269-47bb-a608-4fd898cda285
# ╠═cefe439a-857f-46b2-80b9-ffaf7c53ae37
# ╟─aadb0624-b33a-4ab2-a648-dd388d2fa43f
# ╠═21b9b85e-ca6a-4f78-9272-e4dc5fa0fe36
# ╠═b2c89833-2840-4471-b023-2fa22f8b20bf
# ╠═ba7adbb4-f816-4f48-8705-458213163e3c
# ╟─14b3bf7d-bc78-4c21-a75e-eae7c3465e9f
# ╟─38217246-b987-4693-9a25-c5b88bbb957b
# ╠═a69e711b-bea8-4f9e-8975-585a412728f3
# ╠═6568d8cf-d572-4cea-9929-0a387684f1c8
# ╟─f58c2ce0-8707-4899-8440-0f40eb9768ba
# ╠═9ba12ded-ca90-48e1-a347-017885bb10c0
# ╠═d6669a2b-c5f4-48d0-9127-721751d400f9
# ╠═adf2d25c-95b5-48a9-8d3c-f7268e5ae724
# ╠═fbd4846c-3ab7-42a7-982b-91e4b66bd99b
# ╟─4afa5161-28a6-4528-a50f-90fc02960cd6
# ╠═b47d19f6-b0b1-424f-85ec-6fb65ebe32c2
# ╠═80d2da5f-ca1c-45ff-95f3-102c75ca6b47
# ╠═23fa80bd-b3c0-477b-9afb-5098b9415a0a
# ╟─1a442c43-d1a1-485e-b00a-bcf689a323a4
# ╠═2381e101-b291-4d35-9102-776633a4ddc9
# ╟─d087317a-9f95-4748-85cd-fa224d964e01
# ╠═8729d29e-8d14-4278-82d6-9ec90f2f07d4
# ╠═be3c6b62-7657-4b59-adcb-eacd1ba1d094
# ╠═b6c50e8d-b7a1-4168-a59b-10e4cee23a87
# ╟─4e288174-c167-49ea-8539-1431e1d6964d
# ╠═0e06448f-753d-4ed9-8b17-5454bd3492d3
# ╠═fe9f265d-d65b-4ada-908e-a563f623d896
# ╠═1660b021-21de-4501-b486-562482bd2922
# ╠═3d275a41-e6a5-48bb-9818-20692615499d
# ╠═0f102029-3e59-4f40-a938-0cf9c8c39814
# ╟─91f0626e-470e-4b4b-87e9-45a92b77dcd1
# ╠═cad3c198-bd9e-4da2-aeee-66d2fe25f795
# ╟─b416a8db-dae0-4dcd-9197-4ee517c9d793
# ╠═7cb05d98-764a-4391-a89f-6a855163f6f0
# ╟─49e8fc18-ccdc-4ebe-8cb7-30379b25a969
# ╠═12be5e6a-71bc-4980-82ac-2f011b5a20ea
# ╟─e646d365-3899-4dd6-b4cf-8fad6bac2acf
# ╠═173768ba-2f79-4baf-a8aa-0846c67b7793
# ╟─6d8b731a-f177-4b51-91d4-ff65bc22c8fd
# ╠═4e3e6e46-c396-47fd-a16b-44a566dcf608
# ╟─2b4084e5-f059-4029-b524-080a6eb5dd28
# ╠═186ca4a4-098d-4831-89cc-8400afa6c9dc
# ╠═72c02cb5-0bdf-4ea2-99fd-abbc2be3c27f
# ╠═73fad540-f513-4304-9e33-040fc233ce38
# ╟─af5df3bc-c292-45e6-b340-6efce2c296d5
# ╠═cb351292-c125-4000-86b9-06829d8e5d7c
# ╠═7a8d663c-8db3-4a81-9c03-76b98986d685
# ╠═b3c1fdf5-a298-4398-8a26-61f81ef8404e
# ╠═0aabb462-fc48-49da-88a6-6305ffd1747c
# ╠═b9e3ea9e-e62d-4d26-94b4-eef458db9c93
# ╠═844bb0d4-0cd6-48c0-b948-a61114e6263f
# ╟─172e4c70-ab43-47ef-9721-e33219400eba
# ╠═758be42b-d2d8-41ef-8ad4-a54150f9fe93
# ╠═dfd72215-d461-4f0d-beaa-6a362b762139
# ╠═1330b912-214f-4ef6-b794-a8be6465fedc
# ╠═c50984d1-deaf-49b2-804e-b0ba01716fa7
# ╠═a268b5df-fb29-492a-bf44-922774d3a5fe
# ╠═bb0e6eb9-9361-4f8e-b983-7b8c03b206e1
# ╠═2c178473-6618-4b50-9284-bfcdf448c1d2
# ╠═8901f8c4-420a-4ed2-9740-6c82d9f86cf4
# ╟─1c84c6e8-d34c-474b-8275-266ddd5c3fc3
# ╟─9dd21e83-b280-4589-9041-980382538113
# ╠═ef3bbba2-694c-4054-b023-fb541f1e3159
# ╟─5ff4d10a-f683-4841-8f1a-cb1248d9f660
# ╟─9fc31e25-4cec-4ad8-83a5-9db9a86efbf2
# ╟─b9058e4b-9a98-4847-9240-4b2b2ed3d410
# ╠═e68912e1-68db-41ba-9543-9c44b199b2ff
# ╠═00dc7252-f661-491c-af28-45edc1e861fc
# ╠═010d94d0-e6c3-456a-8303-66f5f5997048
# ╟─0cc43cc3-2e36-442d-acaa-ed0e628ab82c
# ╠═d38485e0-bf7b-4b82-b1a3-4edd13dd35cc
# ╠═00e84e78-07b9-4a5d-9a03-219b4e06e52b
# ╟─a13efe5f-fe95-4d28-84be-63cadf64e356
# ╟─2884b072-cddc-4441-938a-3e4953814b26
# ╟─0a4032eb-d8a1-444c-91d0-ac553cc0dc99
# ╟─e8044aad-4832-4ff4-be1b-c7892acafda0
# ╟─78ab85a9-4b07-4464-9749-6a0de5f45863
# ╟─b7429454-b090-4014-b713-d3434f74059c
# ╠═04eb47bd-1888-43ac-8955-1db2a597b5fc
# ╟─203a21ee-dd8e-4eaf-bae3-3d17fb4bdddc
# ╟─ab42779d-5adf-4a64-a4b4-b4f2ae25ec3c
# ╟─430a683b-b457-45c1-b18d-49f240021a34
# ╠═b1f23586-a8b7-4cec-9413-bac8c4cbabc5
# ╠═7760b280-ee74-4453-ae92-bd9a7b0be230
# ╠═8b505562-6c75-4476-9767-2afca19f2e42
# ╠═4f515f9c-bed7-4b15-946d-551712efacff
# ╠═a99b80b2-589b-441d-81d6-40cc8a910364
# ╠═cd142fca-e547-4814-9c93-4c79097d3c6b
# ╠═b9c571f3-89bc-4590-862a-3abd7b04d8b0
# ╠═f5adb716-bf25-42f3-a0c5-fa5fb2fecf2c
# ╠═a50c7e75-5fd2-4634-9e24-d1855266c8a6
# ╠═1923f292-08de-4b05-8404-dd0a8da603f2
# ╠═0535599d-9fa5-4890-8f7a-46ff6fb9dce7
# ╠═2634bdaa-495f-456d-a4a5-727e0da5c3cb
# ╠═10d5560d-30f1-4e4e-8773-f79d977db866
# ╠═31a4dffe-46bf-4e06-8b44-c625e6ee5b56
