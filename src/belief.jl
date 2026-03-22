@with_kw mutable struct MuonBeliefUpdater <: POMDPs.Updater
	pomdp::MuonPOMDP
	n::Int
	n_pcs::Int
	σ_abc::Float64 = 0.1
	ds0 = missing
end

MuonBeliefUpdater(pomdp::MuonPOMDP, n::Int, n_pcs::Int) = MuonBeliefUpdater(; pomdp, n, n_pcs)

function POMDPs.initialize_belief(up::MuonBeliefUpdater, ds0::Vector{MuonState})
	up.ds0 = ds0
    return sample(ds0, up.n, replace=false)
end

function POMDPs.update(up::MuonBeliefUpdater, b::Vector{MuonState}, a::MuonAction, o::MuonObservations)
    # TODO: transition each state b′ = [transition(s, a) for s in b]
    mp = up.pomdp.mp
	VERBOSE = false
	bk = map(s->s.index, b)
	pcs = 1:up.n_pcs
	VERBOSE && @info length(unique(bk))
    b = deepcopy(b)
    for s in unique(b)
        push!(s.actions, a)
    end
	W = []
	for oi in eachindex(o)
		obs = o[oi]
		sensor_num = obs.sensor_num
		VERBOSE && @info sensor_num, length(unique(bk))
		d_ensemble = up.pomdp.prior["$sensor_num"]
		d_pca, d_pcscr = pca(d_ensemble[bk]; n_components=up.n_pcs)
		try
			ô = observation_weight(mp.dtrue_muon, d_pca, pcs, sensor_num)
			𝐰 = reweight(d_pcscr, pcs, ô; σ=up.σ_abc)
			push!(W, 𝐰)
            # b = sample(b, StatsBase.Weights(𝐰), length(b), replace=true)
		catch err
            if hasproperty(err, :T) && err.T == MuonPOMDPs.np.linalg.LinAlgError
			    VERBOSE && @info "Belief update KDE error when using similar KDE values (sensor_num = $sensor_num)."
            else
                rethrow(err)
            end
		end
	end
	𝐰 = mean(W)
	𝐰 = 𝐰 ./ sum(𝐰)
	b = sample(b, StatsBase.Weights(𝐰), length(b), replace=true)
	@info 𝐰
	@info length(unique(b))
	# b = perturb(b)
	b = reinvigorate(up, b)
	return b
end

function observation_weight(truth, d_pca, pcs, sensor_num)
	K = collect(keys(sort(truth)))
	dtrue_data = truth[K[sensor_num]]
	D = reshape(dtrue_data, 1, :) # (n_samples, n_features)
	Dp = d_pca.transform(D)
    return Dp
end

weight_iqm(𝐱₁, 𝐱₂; c=1) = (1 + ((norm(𝐱₁ - 𝐱₂))^2)/c^2)^(-1/2)
weight_exp(𝐱₁, 𝐱₂; ℓ=1.0) = exp(-norm(𝐱₁ - 𝐱₂) / ℓ)

function reweight(d_pcscr, pcs, ô; σ=missing, c=1000, ℓ=10_000.0, iqm=false)
	if iqm
		𝐰 = [weight_iqm(d_pcscr[d1,pcs], ô[pcs]; c) for d1 in axes(d_pcscr,1)]
	else
		𝐰 = [weight_exp(d_pcscr[d1,pcs], ô[pcs]; ℓ) for d1 in axes(d_pcscr,1)]
	end
	return 𝐰
end

function perturb(b::Vector{MuonState})
	return map(s->MuonState(perturb(s.data), s.index, s.actions), b)
end

function perturb(s)
	f = rand() < 0.5 ? minpool : maxpool
	return f(s, (3,); pad=1, stride=1)
end

function reinvigorate(up::MuonBeliefUpdater, b)
	bu = unique(b)
	nb = length(bu)
	n = up.n
	if nb/n < 1/4 # TODO: Parameterize
		m = n - nb
		@info "Reinvigorating with $m particles..."
		b⁺ = sample(up.ds0, m; replace=true)
		A = first(b).actions
		for i in eachindex(b⁺)
			b⁺[i] = MuonState(b⁺[i].data, b⁺[i].index, deepcopy(A))
		end
		return vcat(bu, b⁺)
	else
		return b
	end
end
