nanmaximum(X) = maximum(x->isnan(x) ? -Inf : x, X)
nanmean(X) = mean(x->isnan(x) ? 0 : x, X)

plot_slice(s::MuonState; kwargs...) = plot_slice(state_data(s); kwargs...)
function plot_slice(data; mode=nanmean, dims=3, c=:viridis, from_slices=false, dpi=1000)
	Z = mapslices(mode, data; dims)
	Z = dropdims(Z, dims = (findall(size(Z) .== 1)...,))
	Plots.heatmap(Z', c=c, ratio=1, dpi=dpi)
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
	return Plots.plot!(xlabel=xlab[dims], ylabel=ylab[dims], dpi=dpi)
end

function plot_slices(data; mode=nanmean, c=:viridis, title="", dpi=1000)
	plts = []
	for dims in [3, 1, 2]
		push!(plts, plot_slice(data; mode, dims, c, from_slices=isempty(title), dpi))
	end
	return Plots.plot(plts...;
		layout=(1,3), size=(600,200), colorbar=false, plot_title=title,
		topmargin=5Plots.mm, bottommargin=5Plots.mm,
		plot_titlefontsize=11, dpi=dpi)
end
