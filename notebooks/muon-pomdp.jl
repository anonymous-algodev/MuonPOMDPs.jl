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

# ╔═╡ c691602a-1ad5-11ef-279d-9f950fefa944
begin
	using Pkg
	Pkg.develop(path="..")
end

# ╔═╡ 69b80ca2-24b5-4611-9c68-4ae3f5264dab
begin
	using Revise
	using MuonPOMDPs
end

# ╔═╡ 98e9dc8b-cea6-47a8-8850-93648fcc1dd3
using Random

# ╔═╡ 1650ed81-d775-45a2-8be0-ea62739f310a
using PlutoUI; TableOfContents()

# ╔═╡ 51b508df-a38e-4e16-89ce-d32994fc8a1c
using Images

# ╔═╡ 5b37e9ff-3d28-4f68-94bc-40de817650ec
using LocalFilters

# ╔═╡ 41d803b5-52f4-48ce-aef7-942924a45caa
using ImageFiltering

# ╔═╡ 8845edb3-3f52-4f89-92bd-84e8e81fd83c
using Flux

# ╔═╡ be995974-8e43-49ea-adfe-e81cd52d8e7f
using POMDPTools

# ╔═╡ 0350af05-6509-4d22-a310-f86df8364164
using Reel

# ╔═╡ a5a0213e-8b9c-4ed6-87b2-53d36c4bf6f0
md"""
# Muon POMDP
"""

# ╔═╡ ec2925b5-3cb2-457f-b467-5edf1ef2d7ea
x = 10

# ╔═╡ d9d147e0-6823-47bb-b7bf-c2294cbb421e
md"""
## Load data
"""

# ╔═╡ 4a6c7e08-60db-473a-a059-43c13a15a153
begin
	url_ground_truth = "https://www.dropbox.com/scl/fi/yt4d3xb3mj5v00trnhx3u/ground_truth_80x80x60.npy?rlkey=qo6hrni4sidp9nryy234g8zq5&dl=1"
	true_intrusion = load_data(url_ground_truth)
end; md"`true_intrusion`"

# ╔═╡ c6629f6f-c394-4beb-9d50-08a7bfa8d652
begin
	url_topography = "https://www.dropbox.com/scl/fi/lua5fwx9dwejml14u3bxp/topography.npy?rlkey=2mrl452k7u0ubn0vwzfp61gm6&dl=1"
	topography = load_data(url_topography; multiplier=10)
end; md"`topography`"

# ╔═╡ 234e5f54-a0fd-4499-9591-cf8760fe46eb
md"""
# Autoencoder data
"""

# ╔═╡ 4f56f15e-c38c-4a2b-98d1-eee1ba7667be
md"""
# MuonParams
"""

# ╔═╡ d5a7aafc-7b62-4df0-a5f3-de1fe0c3e325
mp = MuonParams(true_intrusion, topography);

# ╔═╡ 686e51ba-6ff6-4c40-bbbd-b962936f9a09
md"""
# POMDP
"""

# ╔═╡ 1f76802a-e038-4945-a03b-5d2a24c6c636
# preprocess(pomdp)

# ╔═╡ 1fd92651-f683-4682-8652-f89fdfea6822
pomdp = MuonPOMDP(mp);

# ╔═╡ 214ee634-28f4-4372-9453-4dc89530afe8
md"""
#### Initial state distribution
"""

# ╔═╡ 99768478-c0e3-4ce1-8cd9-de09431cabef
ds0 = initialstate(pomdp);

# ╔═╡ cb58343b-b7c8-4f50-b399-56589d1cbbf0
begin
	plot_slice(ds0[1].data; mode=nanmaximum)
	MuonPOMDPs.Plots.plot!(xlabel="", ylabel="", cbar=false, axis=false)
end

# ╔═╡ cb2d24e9-b77f-43e9-b5dd-5550df7e7b7a
img = Gray.(rotl90(mapslices(nanmaximum, ds0[1].data; dims=3)[:,:,1]))

# ╔═╡ 7f040dbd-ee71-4836-bbd8-54fe1e089eba
save("state.png", img)

# ╔═╡ bb162d60-0182-40ed-aade-94edd23d4723
for (i,s) in enumerate(ds0)
	data = rotl90(mapslices(nanmaximum, s.data; dims=3)[:,:,1])
	img = Gray.(data)
	save("ensembles/state_$i.png", img)
end

# ╔═╡ afaa8720-0031-4659-b255-1b60ca3a253c
for (i,s) in enumerate(ds0)
	data = s.data
	MuonPOMDPs.npzwrite("ensembles/full_state_$i.npy", data)
end

# ╔═╡ 3c57d540-f628-4919-8f54-2c26ca1c6009
md"""
#### Initial state $s_0$
"""

# ╔═╡ 4e10e42d-f29e-4241-b7d1-f8b68d11d4bb
s0 = rand(ds0);

# ╔═╡ 54f24256-1668-4640-ae48-ed37c0063ab6
s0

# ╔═╡ 65f6d9c3-4921-44ea-9062-66f8184649fa
md"""
#### Belief updater
Likelihood-free PCA particle filtering: $\rm L{\small IEP}$
"""

# ╔═╡ 329e07bf-d6f3-46b2-992e-925e667771ff
up = MuonBeliefUpdater(pomdp, 100, 10);

# ╔═╡ e502e4f1-5581-4be4-8486-0857f09c93bc
md"""
#### Initial belief $b_0$
"""

# ╔═╡ 8b684de5-3a61-4ffe-b5b9-b20d80611ad2
Random.seed!(0); b0 = initialize_belief(up, ds0);

# ╔═╡ 7a8f5e56-4914-4c6c-b2a5-42bfee7f0974
md"""
## Running the POMDP
"""

# ╔═╡ bc100b07-764f-445c-9fec-1aae0c8ca591
xx, yy = MuonPOMDPs.np.meshgrid(MuonPOMDPs.np.linspace(200, 3800, 10), MuonPOMDPs.np.linspace(200, 3800, 10))

# ╔═╡ f503fb40-1cc5-45ea-a520-33314700d415
reshape(yy, :)

# ╔═╡ ec6ac1c3-77c5-408f-bc55-d4ca729fe380
function meshgrid(n, a, b, l) #n= number of grids, L= size of interval
	a = transpose(range(a,b,l))
	b = repeat(a,n,1)
	return b
end

# ╔═╡ 2bbf1a6a-9c49-4e6e-90d7-a7cbbb4c175c
a0 = actions(pomdp)[55];

# ╔═╡ f9686cf3-120a-4994-8b07-679b12faef83
sp, o, r = @gen(:sp, :o, :r)(pomdp, s0, a0);

# ╔═╡ 04be543e-3241-46af-8f9f-317597cb55f4
bp = update(up, b0, a0, o); # TODO: how to use action

# ╔═╡ 50ffff93-cb45-4e79-b11d-0fd2758ad791
o

# ╔═╡ efdeccc6-99c8-4d3e-97a0-07d08395fd0a
md"""
# Plots
"""

# ╔═╡ c7e58b74-b2bb-4fc7-a5b8-779cc9ebef27
MuonPOMDPs.Plots.default(fontfamily="Computer Modern", framestyle=:box)

# ╔═╡ 3a4ec54f-c679-4328-a705-08e9a90ee643
plot_slices(mp.intrusion; mode=nanmaximum, title="True intrusion"); MuonPOMDPs.Plots.savefig("true_intrusion_slices.png"); MuonPOMDPs.Plots.plot!()

# ╔═╡ a9817d55-57e7-4914-87f3-ad925ac9f221
plot_slices(mp.intrusion; title="True intrusion"); MuonPOMDPs.Plots.savefig("true_intrusion_slices_mean.png"); MuonPOMDPs.Plots.plot!()

# ╔═╡ ac62ad4f-2fb9-4004-a20e-f01bd84535b0
plot_slices(state_data(s0); mode=nanmaximum, title="State"); MuonPOMDPs.Plots.savefig("single_state_slices.png"); MuonPOMDPs.Plots.plot!()

# ╔═╡ 88abae87-4d45-456f-8ae1-80015e646165
md"""
## Initial belief $b_0$
"""

# ╔═╡ 84132899-35db-4397-a8f1-7f3972a7a982
plot_slices(mean(state_data(b0)); title="Belief mean"); MuonPOMDPs.Plots.savefig("initial_belief_mean_slices.png"); MuonPOMDPs.Plots.plot!()

# ╔═╡ 17cee2f3-268d-482f-99ba-3f968636471b
plot_slices(std(state_data(b0)); c=:cividis, title="Belief std"); MuonPOMDPs.Plots.savefig("initial_belief_std_slices.png"); MuonPOMDPs.Plots.plot!()

# ╔═╡ 56d74f5b-5936-46c7-8a0c-96b7288feb03
md"""
## Updated belief $b'$
"""

# ╔═╡ 0d29862e-220e-4d76-9df6-e44983bf1430
plot_slices(mean(state_data(bp)); title="Belief mean")

# ╔═╡ 0799d0d0-fb2a-4634-9688-16b93db30f76
plot_slices(std(state_data(bp)); c=:cividis, title="Belief std")

# ╔═╡ d33d3ffd-31b7-44c9-aea7-b4d14a159759
md"""
# Perturb
"""

# ╔═╡ 7f0d56d4-eb4e-456b-aa9a-39713c6e9698
function perturb(s)
	f = rand() > 0.5 ? dilate : erode
	v = rand(1:2)
	return f(s, v)
end

# ╔═╡ 012ac809-befd-47cd-80dc-f5e682cc9d3c
plot_slice(perturb(s0.data); mode=nanmaximum)

# ╔═╡ 213badf1-b0bf-47a4-9ef8-6dcd8425c4b5
plot_slices(dilate(s0.data, 3); mode=nanmaximum)

# ╔═╡ b8192d3a-e237-4fa4-bf80-1baf9f11d3b8
# plot_slice(s0.data; mode=nanmaximum)
plot_slice(begin
	local s = s0.data
	for i in 1:4
		s = dilate(s,2)
	end
	s
end; mode=nanmaximum)

# ╔═╡ 43792118-113d-4068-aa0f-9b94029c49c6
plot_slice(imfilter(s0.data, Kernel.gaussian((1,1,1))) .> 0.1; mode=nanmaximum)

# ╔═╡ c725e589-f13b-411e-b78f-2d65025dc353
s = s0.data;

# ╔═╡ 05acdaae-c009-489d-9015-01b84dcf1460
cf = Flux.convfilter((2,2,2), prod(size(s))=>1)

# ╔═╡ bfd943d2-a92e-406f-a6e4-77b6ef062796
plot_slice(maxpool(s, (3,); pad=1, stride=1); mode=nanmaximum)
# plot_slice(minpool(s, (3,); pad=1, stride=1); mode=nanmaximum)
# plot_slice(s; mode=nanmaximum)

# ╔═╡ 6d327cad-a9d3-428b-9766-84f5b56cf78b
plot_slice(s)

# ╔═╡ b6ae52d9-b3d0-4d3f-8265-cea8aeb5a342
function padarray(matrix, pad)
    padded_matrix = zeros(eltype(matrix), size(matrix, 1) + 2*pad, size(matrix, 2) + 2*pad, size(matrix, 3))
    padded_matrix[pad+1:end-pad, pad+1:end-pad, :] .= matrix
    return padded_matrix
end

# ╔═╡ 7a045b82-31e6-420e-8b9d-0ab3ec01e3fc
function minpool(matrix, kernel; stride=1, pad=0)
    padded_matrix = padarray(matrix, pad)
    out_size = ((size(padded_matrix, 1) - kernel[1]) ÷ stride + 1,
                (size(padded_matrix, 2) - kernel[1]) ÷ stride + 1,
                size(matrix, 3))
    pooled_matrix = fill(Inf, out_size...)

    for k in 1:size(matrix, 3)
        for i in 1:out_size[1]
            for j in 1:out_size[2]
                pooled_matrix[i, j, k] = minimum(padded_matrix[
                    (i-1)*stride+1:(i-1)*stride+kernel[1],
                    (j-1)*stride+1:(j-1)*stride+kernel[1],
                    k
                ])
            end
        end
    end

    return pooled_matrix
end

# ╔═╡ d1333420-582a-4074-b16f-6a15fe47ff5d
md"""
# Random policy
"""

# ╔═╡ fa53a8ba-920c-42f0-870a-ce2a3e26c5ca
policy = RandomPolicy(pomdp);

# ╔═╡ 682b7f85-be81-42b9-a8e2-49067350e139
begin
	Random.seed!(2)
	B = [b0]
	S = [s0]
	A = []
	for (sp,a,r,bp,t) in stepthrough(pomdp, policy, up, b0, s0, "sp,a,r,bp,t", max_steps=10)
		@info "t = $t, action.i = $(a.i)"
		push!(B, bp)
		push!(S, sp)
		push!(A, a)
	end
end

# ╔═╡ b3d644fb-496a-4599-b180-378cf0445d69
plot_slice(mean(map(s->perturb(state_data(s)), B[1])); mode=nanmaximum)

# ╔═╡ 2191fa14-a1eb-4cdf-953a-52b640da2ece
plot_slice(mean(state_data(B[1])); mode=nanmaximum)

# ╔═╡ 7e02053f-a001-4e30-92f2-a934f90f0cb7
function plot_actions(pomdp, data, A; title="True intrusion", kwargs...)
	mp = pomdp.mp
	plot_slice(data; mode=nanmaximum, dims=3)
	MuonPOMDPs.Plots.title!(title)
	for a in A
		x = a.x / mp.h[1]
		y = a.y / mp.h[2]
		MuonPOMDPs.Plots.scatter!([x], [y], ms=2, mark=:square, c=:black, msc=:white, label=false, widen=false)
	end
	MuonPOMDPs.Plots.plot!()
end

# ╔═╡ a2967c5a-cfea-4b83-9537-0550f2368983
plot_actions(pomdp, state_data(b0[6]), [a0])

# ╔═╡ 3f4170bf-e941-4a97-932b-8708aac437a8
plot_actions(pomdp, mp.intrusion, [a0])

# ╔═╡ 36ee8203-5268-45fb-9308-38a9f4e8bd2a
plot_actions(pomdp, mp.intrusion, A)

# ╔═╡ 32a4239d-7539-493f-a309-e82f7273bcdf
@bind bt Slider(1:length(B), show_value=true, default=length(B))

# ╔═╡ f753cf38-0842-4cb7-a6aa-327b621f2010
bend = B[bt];

# ╔═╡ 33098036-d731-4692-8de1-8e0d4c57860b
plot_slices(mean(state_data(bend)); title="(Final) belief mean"); MuonPOMDPs.Plots.savefig("final_belief_mean_slices.png"); MuonPOMDPs.Plots.plot!()

# ╔═╡ ae39b550-7070-408b-8caa-c81b251d7dc7
plot_slices(std(state_data(bend)); c=:cividis, title="(Final) belief std"); MuonPOMDPs.Plots.savefig("final_belief_std_slices.png"); MuonPOMDPs.Plots.plot!()

# ╔═╡ e1b0a29c-ae87-4dc5-9419-c3646158667a
plot_slices(mean(state_data(bend)); mode=nanmaximum, title="(Final) belief mean")

# ╔═╡ 5d4b492b-40a5-4c3d-8c09-3944b83e459e
plot_slices(abs.(mean(state_data(bend)) - mp.intrusion); mode=nanmean, title="\$|b - s_\\textrm{true}|\$"); MuonPOMDPs.Plots.savefig("final_belief_difference.png"); MuonPOMDPs.Plots.plot!()

# ╔═╡ 8c1a7d62-fb52-4938-881c-f2699f23f65e
@bind t Slider(eachindex(B), show_value=true, default=length(B))

# ╔═╡ 0867e8e5-e3cb-460d-9db1-3e8767d80cd3
begin
	local true_massive = sum(filter(!isnan, mp.intrusion))
	local estimated_massive = [mean(map(sum, state_data(b))) for b in B]
	@info abs(estimated_massive[end] - true_massive)
	MuonPOMDPs.Plots.plot(abs.(estimated_massive .- true_massive))
end

# ╔═╡ 8d2e7305-dd76-4b2f-93ed-606d6392f5ce
sum(filter(!isnan, mp.intrusion))

# ╔═╡ bcf28787-039a-4508-8992-b4e34fab45af
function plot_ore_mass_distr(b, truth)
	MuonPOMDPs.Plots.histogram(map(sum, state_data(b)),
		normalize=:probability,
		xlims=(6_000, 20_000), label=false, size=(600,300),
		c="#FFE781",
		xlabel="intrusion mass",
		ylabel="probability",
		title="intrusion mass distribution",
		margin=2MuonPOMDPs.Plots.mm)
	MuonPOMDPs.Plots.vline!([sum(filter(!isnan, truth))], c=:crimson, lw=2, label="truth")
	return MuonPOMDPs.Plots.ylims!(0, MuonPOMDPs.Plots.ylims()[2])
end

# ╔═╡ 42464ac1-1ed3-4ce8-ae98-903f75c94d42
plot_ore_mass_distr(B[t], mp.intrusion); MuonPOMDPs.Plots.savefig("mass_distribution_$t.png"); MuonPOMDPs.Plots.plot!()

# ╔═╡ 2ccff5f9-681c-4c48-8f50-e4a76625eda2
md"""
# Animated GIF
"""

# ╔═╡ 30550aa9-6c4a-4a09-ab8d-52a0e9e3ee37
# ╠═╡ disabled = true
#=╠═╡
begin
	frames = Frames(MIME("image/png"), fps=2)
	
	for t in eachindex(B)
		push!(frames, plot_slices(mean(state_data(B[t])); title="belief mean t=$t"))
	end
	write("belief_update.gif", frames)
	LocalResource("./belief_update.gif")
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─a5a0213e-8b9c-4ed6-87b2-53d36c4bf6f0
# ╠═c691602a-1ad5-11ef-279d-9f950fefa944
# ╠═69b80ca2-24b5-4611-9c68-4ae3f5264dab
# ╠═98e9dc8b-cea6-47a8-8850-93648fcc1dd3
# ╠═1650ed81-d775-45a2-8be0-ea62739f310a
# ╠═ec2925b5-3cb2-457f-b467-5edf1ef2d7ea
# ╟─d9d147e0-6823-47bb-b7bf-c2294cbb421e
# ╟─4a6c7e08-60db-473a-a059-43c13a15a153
# ╟─c6629f6f-c394-4beb-9d50-08a7bfa8d652
# ╟─234e5f54-a0fd-4499-9591-cf8760fe46eb
# ╠═51b508df-a38e-4e16-89ce-d32994fc8a1c
# ╠═cb58343b-b7c8-4f50-b399-56589d1cbbf0
# ╠═cb2d24e9-b77f-43e9-b5dd-5550df7e7b7a
# ╠═bb162d60-0182-40ed-aade-94edd23d4723
# ╠═7f040dbd-ee71-4836-bbd8-54fe1e089eba
# ╠═afaa8720-0031-4659-b255-1b60ca3a253c
# ╟─4f56f15e-c38c-4a2b-98d1-eee1ba7667be
# ╠═d5a7aafc-7b62-4df0-a5f3-de1fe0c3e325
# ╟─686e51ba-6ff6-4c40-bbbd-b962936f9a09
# ╠═1f76802a-e038-4945-a03b-5d2a24c6c636
# ╠═1fd92651-f683-4682-8652-f89fdfea6822
# ╟─214ee634-28f4-4372-9453-4dc89530afe8
# ╠═99768478-c0e3-4ce1-8cd9-de09431cabef
# ╟─3c57d540-f628-4919-8f54-2c26ca1c6009
# ╠═4e10e42d-f29e-4241-b7d1-f8b68d11d4bb
# ╠═54f24256-1668-4640-ae48-ed37c0063ab6
# ╟─65f6d9c3-4921-44ea-9062-66f8184649fa
# ╠═329e07bf-d6f3-46b2-992e-925e667771ff
# ╟─e502e4f1-5581-4be4-8486-0857f09c93bc
# ╠═8b684de5-3a61-4ffe-b5b9-b20d80611ad2
# ╟─7a8f5e56-4914-4c6c-b2a5-42bfee7f0974
# ╠═bc100b07-764f-445c-9fec-1aae0c8ca591
# ╠═f503fb40-1cc5-45ea-a520-33314700d415
# ╠═ec6ac1c3-77c5-408f-bc55-d4ca729fe380
# ╠═2bbf1a6a-9c49-4e6e-90d7-a7cbbb4c175c
# ╠═f9686cf3-120a-4994-8b07-679b12faef83
# ╠═04be543e-3241-46af-8f9f-317597cb55f4
# ╠═a2967c5a-cfea-4b83-9537-0550f2368983
# ╠═3f4170bf-e941-4a97-932b-8708aac437a8
# ╠═50ffff93-cb45-4e79-b11d-0fd2758ad791
# ╟─efdeccc6-99c8-4d3e-97a0-07d08395fd0a
# ╠═c7e58b74-b2bb-4fc7-a5b8-779cc9ebef27
# ╠═3a4ec54f-c679-4328-a705-08e9a90ee643
# ╠═a9817d55-57e7-4914-87f3-ad925ac9f221
# ╠═ac62ad4f-2fb9-4004-a20e-f01bd84535b0
# ╟─88abae87-4d45-456f-8ae1-80015e646165
# ╠═84132899-35db-4397-a8f1-7f3972a7a982
# ╠═17cee2f3-268d-482f-99ba-3f968636471b
# ╟─56d74f5b-5936-46c7-8a0c-96b7288feb03
# ╠═0d29862e-220e-4d76-9df6-e44983bf1430
# ╠═0799d0d0-fb2a-4634-9688-16b93db30f76
# ╟─d33d3ffd-31b7-44c9-aea7-b4d14a159759
# ╠═7f0d56d4-eb4e-456b-aa9a-39713c6e9698
# ╠═b3d644fb-496a-4599-b180-378cf0445d69
# ╠═2191fa14-a1eb-4cdf-953a-52b640da2ece
# ╠═012ac809-befd-47cd-80dc-f5e682cc9d3c
# ╠═5b37e9ff-3d28-4f68-94bc-40de817650ec
# ╠═213badf1-b0bf-47a4-9ef8-6dcd8425c4b5
# ╠═b8192d3a-e237-4fa4-bf80-1baf9f11d3b8
# ╠═41d803b5-52f4-48ce-aef7-942924a45caa
# ╠═43792118-113d-4068-aa0f-9b94029c49c6
# ╠═8845edb3-3f52-4f89-92bd-84e8e81fd83c
# ╠═c725e589-f13b-411e-b78f-2d65025dc353
# ╠═05acdaae-c009-489d-9015-01b84dcf1460
# ╠═bfd943d2-a92e-406f-a6e4-77b6ef062796
# ╠═6d327cad-a9d3-428b-9766-84f5b56cf78b
# ╠═b6ae52d9-b3d0-4d3f-8265-cea8aeb5a342
# ╠═7a045b82-31e6-420e-8b9d-0ab3ec01e3fc
# ╟─d1333420-582a-4074-b16f-6a15fe47ff5d
# ╠═be995974-8e43-49ea-adfe-e81cd52d8e7f
# ╠═fa53a8ba-920c-42f0-870a-ce2a3e26c5ca
# ╠═682b7f85-be81-42b9-a8e2-49067350e139
# ╠═7e02053f-a001-4e30-92f2-a934f90f0cb7
# ╠═36ee8203-5268-45fb-9308-38a9f4e8bd2a
# ╠═33098036-d731-4692-8de1-8e0d4c57860b
# ╠═ae39b550-7070-408b-8caa-c81b251d7dc7
# ╠═32a4239d-7539-493f-a309-e82f7273bcdf
# ╠═f753cf38-0842-4cb7-a6aa-327b621f2010
# ╠═e1b0a29c-ae87-4dc5-9419-c3646158667a
# ╠═5d4b492b-40a5-4c3d-8c09-3944b83e459e
# ╠═8c1a7d62-fb52-4938-881c-f2699f23f65e
# ╠═42464ac1-1ed3-4ce8-ae98-903f75c94d42
# ╠═0867e8e5-e3cb-460d-9db1-3e8767d80cd3
# ╠═8d2e7305-dd76-4b2f-93ed-606d6392f5ce
# ╠═bcf28787-039a-4508-8992-b4e34fab45af
# ╟─2ccff5f9-681c-4c48-8f50-e4a76625eda2
# ╠═0350af05-6509-4d22-a310-f86df8364164
# ╠═30550aa9-6c4a-4a09-ab8d-52a0e9e3ee37
