flatten_columns(A) = py"np.array($A).flatten(order='F')"
np_reshape(X, dims...) = permutedims(reshape(X',reverse(dims)...), reverse(1:length(dims))) # https://discourse.julialang.org/t/converting-python-numpy-program-to-julia/4679/7

function download_data(url::String, filename::String)
    res = HTTP.get(url)
    write(filename, res.body)
    return filename
end

function load_data(url::String; multiplier=1, filename="tmp.npy")
    res = HTTP.get(url)
    write(filename, res.body)
    return npzread(filename) .* multiplier
end

function load_ensemble_matrix(filename=joinpath(@__DIR__, "..", "data", "ensemble.npy"), 
        url="https://www.dropbox.com/scl/fi/83nliurps84ifjs9inrv5/results_ensmeble_07022024.npy?rlkey=m3utsjmzhyvqocgimdjteiah5&dl=1")
    if isfile(filename)
        @info "Loading ensembles from cache."
        m_ensemble = npzread(filename)
    else
        m_ensemble = load_data(url; filename)
    end
    m_ensemble = (m_ensemble .≥ 0) * 1
    return m_ensemble
end

function download_ensemble_jsons(url="https://www.dropbox.com/scl/fo/9xd7yzxc8ndgsrxqgtycm/ADBfX_mkAhsR2EGmJ-0aslg?rlkey=l11dy6yqyojlhznr5platcllz&dl=1")
	download_data(url, "muon.zip")
	run(`unzip muon.zip -d muon`)
    mv("muon", joinpath(@__DIR__, "..", "data", "muon"); force=true)
    return nothing
end
