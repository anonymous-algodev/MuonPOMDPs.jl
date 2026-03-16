@with_kw mutable struct MuonParams
    intrusion::Array
    topo::Matrix
    topo_mask::Array
    dtrue_muon::Dict = Dict()
    is_active::BitVector = BitVector()
    muon_sim = missing
    x0 = [0, 0, -480.0]
    h = [50.0, 50.0, 10.0] # meters
    n = size(intrusion)
    xbh = reshape(np.meshgrid(np.linspace(200, 3800, 10), np.linspace(200, 3800, 10))[1], :)
    ybh = reshape(np.meshgrid(np.linspace(200, 3800, 10), np.linspace(200, 3800, 10))[2], :)
    zbh = missing
    dips = 180 .+ fill(60.0, length(xbh))
    lengths = fill(1100, 10*10) # [950.0, 1050.0, 1100.0, 1100.0, 1150.0]
    azimuth = 90.0
    bhs = missing
end

function MuonParams(intrusion::Array, topo::Matrix)
    topo_mask = deepcopy(intrusion)
	topo_mask[topo_mask .== 0] .= true
	topo_mask[topo_mask .== 1] .= true

    mp = MuonParams(intrusion=intrusion, topo=topo, topo_mask=topo_mask)
    boreholes!(mp)

	rho_true = deepcopy(mp.intrusion)
	rho_true[rho_true .== 1] .= 1.5
	rho_true[rho_true .== 0] .= 1.0

    data_dir = joinpath(@__DIR__, "..", "data")
    filename_muon_sim = joinpath(data_dir, "muon_sim.bson")
    filename_is_active = joinpath(data_dir, "is_active.bson")
    filename_dtrue_muon = joinpath(data_dir, "dtrue_muon.bson")
    if isfile(filename_muon_sim) && isfile(filename_is_active) && isfile(filename_dtrue_muon)
        @info "Loading muon data from cache."
        mp.muon_sim = BSON.load(filename_muon_sim)[:muon_sim]
        mp.is_active = BSON.load(filename_is_active)[:is_active]
        mp.dtrue_muon = BSON.load(filename_dtrue_muon)[:dtrue_muon]
    else
        mp.muon_sim, mp.is_active = forward(mp)
    	mp.dtrue_muon = mp.muon_sim.get_data(flatten_columns(rho_true)[mp.is_active])
    end
    return mp
end
