function muon_data_keys()
    # name muon observation data as var d_muon
    d_dir = joinpath(@__DIR__, "..", "data", "muon")
    if !isdir(d_dir)
        download_ensemble_jsons()
    end
    d_fname = joinpath(d_dir, "100holes_real_")
    d_muon = open("$(d_fname)0.json", "r") do f
        JSON.parse(f)
    end
    return sort(collect(keys(d_muon)))
end

function muon_data(n_reals::Int; sensor_num=6)
    d_muon_keys = muon_data_keys()
    K = d_muon_keys[sensor_num]
    
    # name muon observation data as var d_muon
    d_dir = joinpath(@__DIR__, "..", "data", "muon")
    d_fname = joinpath(d_dir, "100holes_real_")
	d_ensemble = []
	for i in (1:n_reals) .- 1
		loaded_dicts = open("$(d_fname)$i.json", "r") do f
			JSON.parse(f)
		end
		push!(d_ensemble, loaded_dicts[K])
	end
    return d_ensemble
end

function muon_pca(n_reals::Int; n_components=100)
    d_ensemble = muon_data(n_reals)
    return pca(d_ensemble; n_components=n_components)
end

"""
`n_components`: number of PCA components (bases)
`sensor_num`: extract muon observation data for each sensor.
"""
function pca(data; n_components=100)
    d_pca = py"PCA"(n_components=n_components, svd_solver="full")
    d_pca.fit(data)
    d_pcscr = d_pca.transform(data) # use all the PCs
    return d_pca, d_pcscr, data
end


function scree_plot(d_pca)
    ax = figure()
    Y = cumsum(d_pca.explained_variance_ratio_)
    plot(eachindex(Y) .- 1, Y)
    title("Scree plot")
    ylabel("Cumulative var")
    xlabel("n_PC")
    return ax
end


function plot_pca(d_pca, d_pcscr; retain_var=98, pcnum=1)
    Y = cumsum(d_pca.explained_variance_ratio_)
    d_keepPCs = argmin(abs.(Y .- retain_var*0.01))
    d_pcscr = d_pcscr[:, 1:d_keepPCs-1]

    ax = figure(figsize=(5,5))
    scatter(d_pcscr[:,pcnum], d_pcscr[:,pcnum+1], alpha=0.7, 
                s=40, facecolor="w",
                # vmax=0.08, 
                edgecolors="k", linewidth=1)
    xlabel("Muon data PC $pcnum", fontsize=15)
    ylabel("Muon data PC $(pcnum+1)", fontsize=15)
    title("Scatter plot of Muon data PCs for all the 400 realizations")
    return ax
end