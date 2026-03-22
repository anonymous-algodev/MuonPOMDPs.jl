struct MuonState
    data::Array{Int,3}
    index::Int
    actions::Vector
end

MuonState(data::Array{Int,3}, index::Int) = MuonState(data, index, [])

struct MuonAction
    x
    y
    i
    keys
end

Base.hash(a::MuonAction, h::UInt) = hash(Tuple(getproperty(a, p) for p in propertynames(a)), h)
Base.isequal(a1::MuonAction, a2::MuonAction) = all(isequal(getproperty(a1, p), getproperty(a2, p)) for p in propertynames(a1))
Base.:(==)(a1::MuonAction, a2::MuonAction) = isequal(a1, a2)

struct MuonObservation
    data
    sensor_num::Int
end

const MuonObservations = Vector{MuonObservation}

state_data(s::MuonState) = s.data
state_data(S::Vector{MuonState}) = map(state_data, S)

function load_prior(n_realizations::Int, sensor_keys::Vector)
    data_dir = joinpath(@__DIR__, "..", "data")
    filename_prior = joinpath(data_dir, "prior.jld2")
    if isfile(filename_prior)
        @info "Loading prior data from cache."
        prior = load(filename_prior)
        return prior
    else
        @info "Generating prior data..."
        prior = Dict()
        K = eachindex(sensor_keys)
        for (i,sensor_num) in enumerate(K)
            @info "$i/$(length(K))"
            prior["$sensor_num"] = muon_data(n_realizations; sensor_num)
        end
        return prior
        # return Dict(sensor_num=>muon_data(n_realizations; sensor_num) for sensor_num in eachindex(sensor_keys))
    end
end

@with_kw mutable struct MuonPOMDP <: POMDP{MuonState,MuonAction,MuonObservations}
    mp::MuonParams
    m_ensemble::Matrix = load_ensemble_matrix() # TODO: How to generate ensemble GIVEN an action?
    n_realizations::Int = size(m_ensemble, 1) # total number of prior model realizations
    sensor_keys::Vector = muon_data_keys() # number of sensors for a single prior
    prior::Dict = load_prior(n_realizations, sensor_keys)
    drill_every::Int = 1
    discount::Float64 = 0.99
end

MuonPOMDP(mp::MuonParams) = MuonPOMDP(mp=mp)

function MuonObservation(pomdp::MuonPOMDP, sensor_num::Int)
    truth = pomdp.mp.dtrue_muon
    K = collect(keys(sort(truth)))
    dtrue_data = truth[K[sensor_num]]
    return MuonObservation(dtrue_data, sensor_num)
end

get_state(pomdp::MuonPOMDP, i::Int) = MuonState(get_state(pomdp.mp, pomdp.m_ensemble, i), i)

function POMDPs.initialstate(pomdp::MuonPOMDP)
    return [get_state(pomdp, i) for i in 1:pomdp.n_realizations]
end

function sensorindex2keys(pomdp, sensor_idx)
	K = pomdp.sensor_keys
	I = findall(k->occursin("$(sensor_idx-1)_", k), K)
	return K[I]
end

function sensorkeys2nums(pomdp::MuonPOMDP, keys)
	return map(k->findfirst(pomdp.sensor_keys .== k), keys)
end

function POMDPs.actions(pomdp::MuonPOMDP, s::Union{Missing,MuonState}=missing)
    # TODO: Unrestricted action space
    # h = pomdp.mp.h
    # every = pomdp.drill_every
    # x_dim, y_dim, z_dim = pomdp.mp.n
    # A = [[x,y] .* h[1:2]  for x in 1:every:x_dim, y in 1:every:y_dim]

    # Ax = [200, 1400, 2100, 2600, 3800]
    # Ay = fill(2500, length(Ax))
    Ax = pomdp.mp.xbh # collect(range(200, 3800, 10))
    Ay = pomdp.mp.ybh # collect(range(200, 3800, 10))
    if ismissing(s)
        A = [MuonAction(ax, ay, i, sensorindex2keys(pomdp, i)) for (i,(ax,ay)) in enumerate(zip(Ax,Ay))]
    else
        A = []
        Ai = unique(map(a->a.i, s.actions))
        for (i,(ax,ay)) in enumerate(zip(Ax,Ay))
            if i ∉ Ai
                a = MuonAction(ax, ay, i, sensorindex2keys(pomdp, i))
                push!(A, a)
            end
        end
    end
    # if !ismissing(s)
    #     # Filter already selected boreholes.
    #     A = filter(a->a ∉ s.actions, A)
    # end
    return A
end

function POMDPs.actions(pomdp::MuonPOMDP, b::Vector{MuonState})
    return union([actions(pomdp, s) for s in b]...)
end

function POMDPs.gen(pomdp::MuonPOMDP, s::MuonState, a::MuonAction, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
	sp = MuonState(s.data, s.index, vcat(s.actions, a))
	r = POMDPs.reward(pomdp, s, a)
	sensor_nums = sensorkeys2nums(pomdp, a.keys)
	o = [MuonObservation(pomdp, sensor_num) for sensor_num in sensor_nums]
	return (; sp, o, r)
end

POMDPs.discount(pomdp::MuonPOMDP) = pomdp.discount

function POMDPs.reward(pomdp::MuonPOMDP, s, a)
    # TODO: Implement
    return 0
end

function POMDPs.isterminal(pomdp::MuonPOMDP, s::MuonState)
    return isempty(actions(pomdp, s))
end

function preprocess(pomdp::MuonPOMDP)
    mp = pomdp.mp
    data_dir = joinpath(@__DIR__, "..", "data")
    
    @info "Saving `muon_sim`..."
    muon_sim = mp.muon_sim
    BSON.@save joinpath(data_dir, "muon_sim.bson") muon_sim
    
    @info "Saving `is_active`..."
    is_active = mp.is_active
    BSON.@save joinpath(data_dir, "is_active.bson") is_active

    @info "Saving `dtrue_muon`..."
    dtrue_muon = mp.dtrue_muon
    BSON.@save joinpath(data_dir, "dtrue_muon.bson") dtrue_muon
    
    @info "Saving `prior`..."
    prior = pomdp.prior
    save(joinpath(data_dir, "prior.jld2"), prior)

    return nothing
end