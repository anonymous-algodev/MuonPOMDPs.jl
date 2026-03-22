function plot_orebody_3d(orebody; use_pyplot=false, plt_title="", camera=(30,30))
    if use_pyplot
        rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
        rcParams["font.family"] = "Times New Roman" # default = "DejaVu Sans"

        ax = figure(figsize=(7,7))
        ax.add_subplot(projection="3d")

        m_sdf = skfmm.distance(orebody .- 0.5)
        plt_contour = np.argwhere(m_sdf .< 2 .&& m_sdf .≥ -0.5)
        scatter3D(plt_contour[:,1],
                plt_contour[:,2],
                plt_contour[:,3],
                c="y",
                linewidth=0.1, edgecolor="k",
                marker="s", vmax=1, s=25)
        title(plt_title, fontsize=24)
        xlim(0,size(m_sdf,1))
        ylim(0,size(m_sdf,2))
        zlim(0,size(m_sdf,3))
        xlabel("x-dimension")
        ylabel("y-dimension")
        # zlabel("z-dimension")
        ax.axes[1].xaxis.pane.set_edgecolor("k")
        ax.axes[1].yaxis.pane.set_edgecolor("k")
        ax.axes[1].zaxis.pane.set_edgecolor("k")
        ax.tight_layout()
        return ax
    else
        X = []
        Y = []
        Z = []
        
        for i in axes(orebody,1)
            for j in axes(orebody,2)
                for k in axes(orebody,3)
                    if orebody[i,j,k] == 1
                        push!(X, i)
                        push!(Y, j)
                        push!(Z, k)
                    end
                end
            end
        end
        
        return Plots.scatter(X, Y, Z;
            ms=2,
            xlims=(1,80),
            ylims=(1,80),
            zlims=(1,80),
            c=:gold,
            msc=:black,
            msw=0.1,
            mark=:square,
            camera,
            label=false,
        )
    end
end


"""
Define boreholes, where muon sensors will be placed.
"""
boreholes!(mp::MuonParams) = boreholes!(mp, mp.xbh, mp.ybh)
function boreholes!(mp::MuonParams, xbh::Vector, ybh::Vector)
    topo, x0, h, n = mp.topo, mp.x0, mp.h, mp.n
    mp.xbh = xbh
    mp.ybh = ybh
    mp.zbh = [py"snap_to_topo"((x, y, 0), topo, x0, h, n)[3] for (x,y) in zip(mp.xbh, mp.ybh)]
    mp.bhs = [py"Borehole"((mp.xbh[i], mp.ybh[i], mp.zbh[i]), mp.dips[i], mp.azimuth, mp.lengths[i], i) for i in eachindex(mp.xbh)]
    return mp.bhs
end


"""
Visualize the vertical and horizontal cross-sections.
"""
function plot_sections(mp::MuonParams)
    intrusion, topo_mask = mp.intrusion, mp.topo_mask
    bhs, x0, hx, hz = mp.bhs, mp.x0, mp.h[1], mp.h[3]

    ax = figure(figsize=(15,7))
    ax.add_subplot(131)
    imshow((intrusion .* topo_mask)[:,:,end-11]', origin="lower")
    xticks([0, 39, 79], [0, 2000, 4000])
    yticks([0, 39, 79], [0, 2000, 4000])
    ylabel("Y (m)"), xlabel("X (m)")
    title("Horizontal section")
    subplot(132)
    imshow((intrusion .* topo_mask)[40,:,:]', origin="lower")
    xticks([0, 39, 79], [0, 2000, 4000])
    yticks([0, 20, 40, 60], [-480, -280, -80, 120])
    ylabel("Z (m)"), xlabel("Y (m)")
    title("Vertical section YZ")
    subplot(133)
    img = imshow((intrusion .* topo_mask)[:,40,:]', origin="lower")
    for bh in bhs
        plot((bh.x0[1]-x0[1])/hx, (bh.x0[3]-x0[3])/hz, "r*", markersize=10)
    end
    colorbar(img, ax=gca(), orientation="vertical")
    xticks([0, 39, 79], [0, 2000, 4000])
    yticks([0,20, 40, 60], [-480, -280, -80, 120])
    ylabel("Z (m)"), xlabel("X (m)")
    title("Vertical section XZ")
    ax.tight_layout()
    return ax
end


"""
Visualize ground truth and topography in 3D.
"""
function plot_ground_truth(mp::MuonParams)
    topo = mp.topo
    bhs, x0, h, n, lengths = mp.bhs, mp.x0, mp.h, mp.n, mp.lengths

    top_x = np.argwhere(topo)[:,1]
    top_y = np.argwhere(topo)[:,2]
    top_z = topo[unique(top_x .+ 1), unique(top_y .+ 1)]
    # create x, y, z array
    # top_xyz = (top_x[1:end-1] .* 50 .+ 25, top_y[1:end-1] .* 50 .+ 25, top_z[1:end-1]) # use cell center only

    m_sdf = skfmm.distance(mp.intrusion .- 0.5)
    plt_contour = np.argwhere(m_sdf .< 2 .&& m_sdf .≥ -0.5)

    fig = plt.figure(figsize=(7,7))
    fig.add_subplot(projection="3d")

    scatter3D(plt_contour[:,1],
              plt_contour[:,2],
              plt_contour[:,3] .- 48,
              c = "y",
              linewidth=0.1, edgecolor="k",
              marker="s", vmax=1,
              s=25)
    scatter3D(top_x[1:2:end],
              top_y[1:2:end],
              (top_z ./ 10)[1:2:end],
              c = "k",
              linewidth=0.15, edgecolor="k",
              marker="s", alpha=0.4,
              s=5)

    # Plot boreholes and sensor locations
    for (ibh, rij) in enumerate(bhs)
        ix0, iy0, iz0 = py"get_cell_index"(rij.x0, x0, h, n)
        @show rij(lengths[ibh])
        ix1, iy1, iz1 = py"get_cell_index"(rij(lengths[ibh]), x0, h, n)
        plot([ix0, ix1], [iy0, iy1], [iz0-48, iz1-48], linewidth=4, c="m")
        n_sensors_per_bh = 9
        for ti in np.linspace(150.0, lengths[ibh]-2.5*h[3], n_sensors_per_bh)
            loc = rij(ti) - x0
            scatter3D(loc[1]/h[1], loc[2]/h[2], loc[3]/h[3]-48, c="g", marker="o", s=10)
        end
    end

    xlim(0, size(m_sdf,1))
    ylim(0, size(m_sdf, 2))
    zlim(-48, 13)
    xlabel("X dim (×50m)", fontsize = 15)
    ylabel("Y dim (×50m)", fontsize = 15)
    zlabel("Z dim (×10m)", fontsize = 15)
    fig.axes[1].xaxis.pane.set_edgecolor("k")
    fig.axes[1].yaxis.pane.set_edgecolor("k")
    fig.axes[1].zaxis.pane.set_edgecolor("k")

    # same y, different x (vertical boreholes, not angels (viz. component))
    return fig
end


function plot_state_action(mp::MuonParams, s::Array{<:Real, 3})
    bhs, x0, h, n, lengths = mp.bhs, mp.x0, mp.h, mp.n, mp.lengths

    m_sdf = skfmm.distance(s .- 0.5)
    plt_contour = np.argwhere(m_sdf .< 2 .&& m_sdf .≥ -0.5)

    fig = plt.figure(figsize=(7,7))
    fig.add_subplot(projection="3d")

    scatter3D(plt_contour[:,1],
              plt_contour[:,2],
              plt_contour[:,3] .- 48,
              c = "y",
              linewidth=0.1, edgecolor="k",
              marker="s", vmax=1,
              s=25)

    # Plot boreholes and sensor locations
    for (ibh, rij) in enumerate(bhs)
        ix0, iy0, iz0 = py"get_cell_index"(rij.x0, x0, h, n)
        @show rij(lengths[ibh])
        ix1, iy1, iz1 = py"get_cell_index"(rij(lengths[ibh]), x0, h, n)
        plot([ix0, ix1], [iy0, iy1], [iz0-48, iz1-48], linewidth=4, c="m")
        n_sensors_per_bh = 9
        for ti in np.linspace(150.0, lengths[ibh]-2.5*h[3], n_sensors_per_bh)
            loc = rij(ti) - x0
            scatter3D(loc[1]/h[1], loc[2]/h[2], loc[3]/h[3]-48, c="g", marker="o", s=10)
        end
    end

    xlim(0, size(m_sdf,1))
    ylim(0, size(m_sdf, 2))
    zlim(-48, 13)
    xlabel("X dim (×50m)", fontsize = 15)
    ylabel("Y dim (×50m)", fontsize = 15)
    zlabel("Z dim (×10m)", fontsize = 15)
    fig.axes[1].xaxis.pane.set_edgecolor("k")
    fig.axes[1].yaxis.pane.set_edgecolor("k")
    fig.axes[1].zaxis.pane.set_edgecolor("k")

    # same y, different x (vertical boreholes, not angels (viz. component))
    return fig
end


function get_mesh(mp::MuonParams)
    x0, h, n = mp.x0, mp.h, mp.n
    return py"TensorMesh"([fill(hi, ni) for (hi,ni) in zip(h,n)], x0=x0)
end


function forward(mp::MuonParams)
    # Create a discretize tensor mesh
    h, bhs, lengths = mp.h, mp.bhs, mp.lengths
    mesh = get_mesh(mp)

    # Create radiograph grid specifying the grid of ray directions
    # along which opacities will be computed.
    tan_theta_max = np.tan(np.radians(60))
    nx = 20
    ny = 20
    xgrid = np.linspace(-tan_theta_max, tan_theta_max, nx)
    ygrid = np.linspace(-tan_theta_max, tan_theta_max, ny)

    # Create muon sensors
    sensors = OrderedDict()
    n_sensors_per_bh = 9
    sensor_locs = zeros(length(bhs)*n_sensors_per_bh, 3)
    for (ibh, bh) in enumerate(bhs)
        for (i,ti) in enumerate(np.linspace(150.0, lengths[ibh]-2.5*h[3], n_sensors_per_bh))
            loc = bh(ti)
            sensor_locs[(bh.id-1)*n_sensors_per_bh + i, :] = loc
            sensors["$(bh.id)_$i"] = py"MuonSensor"(loc, xgrid, ygrid)
        end
    end

    # Define active cell mapping, for later use in SimPEG inversion
    # SimPEG inversion will work on all cells above the lowest sensor and
    # below the topography. Forward operator will be applied to all cells
    # after mapping the active cell model to the full mesh.
    minz = minimum(sensor_locs[:,3])
    gridCC = mesh.gridCC
    is_active = np.logical_not(np.isnan(mp.topo_mask))
    is_active = flatten_columns(is_active)
    is_active = is_active .&& gridCC[:,3] .≥ minz - h[3]
    active_map = py"maps".InjectActiveCells(mesh, is_active, valInactive=0.0)

    # Generate fwd operator
    # We wrap the forward operator in a SimPEG simulation object
    # for use in SimPEG inversion routines. To access the underlying
    # scipy sparse matrix forward operator, use the attribute muon_sim_simpeg._G
    muon_sim_simpeg = py"ToyMuonSimulationSimPeg"(mesh, sensors, model_map=active_map)

    # (SLOW). Run once. Forward operator. Once we have this operator, we just run `simpeg.get_data(...)`
    # Only have to rerun if you change the geometry of the mesh (e.g., 80x80 to 160x160 mesh, changes sensitivity matrix, linear operator)
    return muon_sim_simpeg, is_active
end


function in_prism(p)
	# Define a triangular prism intrusion so we can see the difference in opacity
	# between the null model with no intrusion, a dumb guess at the intrusion shape,
	# and the true model.
	v1 = [1600.0, 20.0]
	v2 = [2700.0, 20.0]
	v3 = [2100.0, -220.0]

	# pt = [2000, 2000.0, -80.0]
	# pt2 = [2000, 2000.0, -450.0]
	prism_xlims = [550.0, 3350.0]

    r1 = py"np.c_[$(v1[2] .- p[:,3]), $(v1[1] .- p[:,2])]"
    r2 = py"np.c_[$(v2[2] .- p[:,3]), $(v2[1] .- p[:,2])]"
    r3 = py"np.c_[$(v3[2] .- p[:,3]), $(v3[1] .- p[:,2])]"
    norm1 = A -> map(norm, eachslice(A, dims=1))
    phi1 = np.arccos(sum(r1 .* r2, dims=2) ./ (norm1(r1) .* norm1(r2)))
    phi2 = np.arccos(sum(r2 .* r3, dims=2) ./ (norm1(r2) .* norm1(r3)))
    phi3 = np.arccos(sum(r3 .* r1, dims=2) ./ (norm1(r3) .* norm1(r1)))
    angle_sums = phi1 + phi2 + phi3
    in_tri = np.isclose(angle_sums, fill(2π, size(phi1)...), atol=1e-3)
    prism_mask = in_tri .&& p[:,1] .≥ prism_xlims[1] .&& p[:,1] .≤ prism_xlims[2]
    return angle_sums, prism_mask
end


function get_rho_prism(mp::MuonParams, mesh)
    intrusion, n = mp.intrusion, mp.n
    gridCC = mesh.gridCC
    angle_sums, prism_mask = in_prism(gridCC)
    prism_mask = reshape(prism_mask, n...)
    rho_prism = zeros(size(intrusion)...)
    rho_prism[.!isnan.(intrusion)] .= 1.0
    rho_prism[prism_mask] .= 1.5
    return rho_prism
end


function plot_rho(mp::MuonParams, rho_prism)
    topo_mask = mp.topo_mask

    ax = figure(figsize=(12,7))
    ax.add_subplot(131)
    imshow((rho_prism .* topo_mask)[:,:,end-11]', origin="lower")
    xticks([0, 39, 79], [0, 2000, 4000])
    yticks([0, 39, 79], [0, 2000, 4000])
    ylabel("Y (m)")
    xlabel("X (m)")
    title("Horizontal section")
    subplot(132)
    imshow((rho_prism .* topo_mask)[40,:,:]', origin="lower")
    xticks([0, 39, 79], [0, 2000, 4000])
    yticks([0, 20, 40, 60], [-480, -280, -80, 120])
    ylabel("Z (m)")
    xlabel("Y (m)")
    title("Vertical section YZ")
    subplot(133)

    img = imshow((rho_prism .* topo_mask)[:,40,:]', origin="lower")
    colorbar(img, ax=gca(), orientation="vertical")
    xticks([0, 39, 79], [0, 2000, 4000])
    yticks([0, 20, 40, 60], [-480, -280, -80, 120])
    ylabel("Z (m)")
    xlabel("X (m)")
    title("Vertical section XZ")
    ax.tight_layout()

    return ax
end


function plot_radiographs(d_muon, op=missing)
    # Plot null model radiographs: (null = empty space, only topography without intrusion inside)
    nx = 20
    ny = 20
    fig, ax = subplots(9,5,figsize=(12, 20))
    for (axi, (data_id, data_arr)) in zip(ax, sort(d_muon))
        if !ismissing(op)
            data_arr = op(data_id)
        end
        img = axi.imshow(reshape(data_arr, nx, ny), cmap="viridis", origin="lower")
        axi.set_title("Radiograph, Sensor $data_id", fontsize=10)
        axi.tick_params(axis="both", which="major", labelsize=8)
        cbar = fig.colorbar(img, ax=axi)
        cbar.ax.tick_params(labelsize=8)
    end
    return fig
end


function inversion(mp::MuonParams, dtrue_muon, std_muon, muon_sim_simpeg, rho0, is_active, mesh)
    # Check that we have the Pardiso solver installed, not strictly necessary but
    # it is much faster than the default LU solver. This will be the sparse-direct
    # linear solver used internally as part of the Hessian search direction solver
    # preconditioning
    # from pymatsolver import Pardiso
    # from pypardiso import PyPardisoSolver as DefaultSolver

    # Define the data misfit function as the L2 norm of the weighted
    # residual between the observed data and the data predicted for a given model.
    true_data_vec = np.concatenate([di for di in values(dtrue_muon)])
    data_std = np.concatenate([si for si in values(std_muon)])
    dobs_vec = true_data_vec .+ randn(length(true_data_vec)) .* data_std
    data_object = py"data".Data(
        py"MuonSurvey"(size(dobs_vec,1)),
        dobs=dobs_vec,
        standard_deviation=data_std
    )
    dmis = py"data_misfit".L2DataMisfit(data=data_object, simulation=muon_sim_simpeg)

    # Define the Tikhonov regularization.
    reg = py"regularization".WeightedLeastSquares(mesh,
        active_cells=is_active,
        mapping=py"None",
        reference_model= rho0[is_active],
        reference_model_in_smooth=false,
        alpha_s=1e-6,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0
    )

    # Define how the optimization problem is solved. Here we will use a projected
    # Gauss-Newton approach that employs the conjugate gradient solver.
    opt = py"optimization".ProjectedGNCG(
        maxIter=10, lower=0.95, upper=2.0, maxIterLS=6, maxIterCG=150, tolCG=1e-4,
        print_type="ubc"
    )

    # Here we define the inverse problem that is to be solved
    inv_prob = py"inverse_problem".BaseInvProblem(dmis, reg, opt, beta=100.0)

    # Configure the inversion using directives

    # Defining the fractional decrease in beta and the number of Gauss-Newton solves
    # for each beta value.
    beta_schedule = py"directives".BetaSchedule(coolingFactor=5, coolingRate=1)

    # Options for outputting recovered models and predicted data for each beta.
    save_iteration = py"directives".SaveOutputEveryIteration(save_txt=false)

    # Setting a stopping criteria for the inversion.
    target_misfit = py"directives".TargetMisfit(chifact=1.0)

    # The directives are defined as a list.
    directives_list = [
        beta_schedule,
        save_iteration,
        target_misfit,
    ]

    # Here we combine the inverse problem and the set of directives
    inv = py"inversion".BaseInversion(inv_prob, directives_list)

    # Run inversion
    @info "Running inversion..."
    run_name = "halfspace-start-w-0p005-noise-chi-1-smoothmod-mod02"
    recovered_model = inv.run(rho0[is_active]) # updated belief

    return recovered_model
end
