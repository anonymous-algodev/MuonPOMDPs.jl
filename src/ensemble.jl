function get_state(mp::MuonParams, m_ensemble, i)
    x_dim, y_dim, z_dim = mp.n    
    return np_reshape(m_ensemble[i,:], x_dim, y_dim, z_dim)
end

function plot_state(s::Array{<:Real, 3}; i_ensemble="", i=1, prev_fig=missing, topdown=false)
    new_fig = ismissing(prev_fig)
    figs = new_fig ? figure(figsize=(4,7)) : prev_fig
    
    # calcuate the signed distance for visualization
    m_sdf = skfmm.distance(s .- 0.5)
    plt_contour = np.argwhere(m_sdf .< 2 .&& m_sdf .≥ -0.5)
    !new_fig && figs.add_subplot(1, 4, i, projection="3d")
    scatter3D(plt_contour[:,1], 
                plt_contour[:,2],
                plt_contour[:,3], 
                c = "y",
                linewidth=0.1, edgecolor="k",
                marker="s", vmax=1,
                s=25)
    suffix = isempty(i_ensemble) ? "" : " $i_ensemble"
    xlabel("x")
    ylabel("y")
    zlabel("z")
    title("realization example$suffix", fontsize=18)
    xlim(10, size(m_sdf,1)-5)
    ylim(5, size(m_sdf,2)-5)
    zlim(5, size(m_sdf,3)-5)
    
    if topdown
        gca().view_init(90, -90)
    end
    
    if !new_fig
        for ax in figs.axes
            ax.set_axis_off()
        end
    end
    if new_fig
        plt.tight_layout()
        return figs
    else
        return nothing
    end
end

function plot_state(mp::MuonParams, m_ensemble, i_ensemble=1, i=1; prev_fig=missing)
    # calcuate the signed distance for visualization
    s = get_state(mp, m_ensemble, i_ensemble)
    return plot_state(s; i_ensemble, i, prev_fig)
end

function plot_ensembles(mp::MuonParams, m_ensemble, n=4)
    figs = figure(figsize=(13,7))
    for i in 1:n
        plot_state(mp, m_ensemble, i*10, i; prev_fig=figs)
    end
    plt.tight_layout()
    return figs
end
