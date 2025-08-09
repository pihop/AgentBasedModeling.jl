### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ fd7efc3c-ff22-11ef-34c6-dfce6f883a42
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="Catalyst"),
        Pkg.PackageSpec(name="ModelingToolkit"),
		Pkg.PackageSpec(name="Distributions"),
		Pkg.PackageSpec(name="CairoMakie"),
		Pkg.PackageSpec(name="ColorSchemes"),
		Pkg.PackageSpec(name="Colors"),
		Pkg.PackageSpec(name="Graphs"),
		Pkg.PackageSpec(name="MetaGraphsNext"),
		Pkg.PackageSpec(name="NetworkLayout"),
		Pkg.PackageSpec(name="GraphMakie"),
		Pkg.PackageSpec(name="StatsBase"),
		Pkg.PackageSpec(name="Interpolations"),
		Pkg.PackageSpec(url="https://github.com/pihop/AgentBasedModeling.jl")
    ])
	using AgentBasedModeling
	using Catalyst
	using ModelingToolkit
	using Distributions
	using LinearAlgebra
	using Interpolations
	using CairoMakie
	using StatsBase
	using ColorSchemes
	using Colors
	using Graphs
	using MetaGraphsNext
	using GraphMakie
	using NetworkLayout
end;

# ╔═╡ ea9c5a6d-7225-43b7-b3f9-562f2185f5b0
md"""
# Cell-to-cell communication example
"""

# ╔═╡ 6b3dab0d-b5c5-4632-a7f2-43fa0a07b01f
md"""
# Cell dynamics
"""

# ╔═╡ 2cdf4d9f-0374-4bb6-bf7f-f34361d1a6b7
md"""
The cell state dynamics descibe the growth of the cell size and autoinducer production.
"""

# ╔═╡ 1279c064-b5ea-4ea7-8f70-097a4ce97490
begin
	@independent_variables t 
	@abm_variables C(t) p(t) τ(t) s(t) Env(t)
	@parameters s0 α kprod K d
	D = Differential(t)
	
	cell_dynamics = @reaction_network begin
	    @parameters α
	    @species p(t)
	    @variables s(t)
	    @equations begin
	        D(p) ~ 0.0
	        D(s) ~ α*s
	    end
	    s*hill(p, kprod, s*K, 2) + d, 0 --> p
	end

	Cell = AgentDynamics(cell_dynamics, (s0, ))
end;

# ╔═╡ 5b42ff0f-ef47-487b-aff5-188b06f13c96
md"""
# Environment dynamics
"""

# ╔═╡ e51ee76f-c8b9-4884-9cc9-7928cd4e0f2f
md"""
The environment agent state is defined to keep a count of the autoinducer molecules in the environment. Dynamics within the environment are not considered here.
"""

# ╔═╡ 978707eb-3ae3-4cdf-b5a0-28b16e2cbc13
Environment = AgentDynamics(EmptyTraitProblem(), (p, ));

# ╔═╡ a8cff812-98ae-42cc-b1de-0130b5326f12
md"""
# Interactions
"""

# ╔═╡ b549d1a5-4e90-417a-9890-7ad682f7f6af
md"""
## Division
"""

# ╔═╡ 03dba07e-a5b9-45cc-9f6e-1572a3bc5fb0
md"""
Hazard function for the gamma distribution. Gammahaz function parametrised by the mean μ and (cv)^2 of the distribution sets the hazard value of the right tail to the theoretical limit 1/(μ*(cv)^2).
"""

# ╔═╡ 0077a4fa-340c-411c-b95e-56a03bb3fb0b
begin
	hazard(d::Gamma, x) = exp(logpdf(d, x) - logccdf(d, x))
	function gammahazard(μ, cv2, x) 
	    iszero(μ) && return 0.0
	    g = Gamma(1/cv2, μ*cv2)
	    x > μ + 4*std(g) && return 1/(μ*cv2)
	    return hazard(g, x)
	end
end;

# ╔═╡ 41f432e8-9049-4b75-83a3-0bbe464f1c52
begin
	# Adder rule for cell division.
	γdivision(μ, cv2, s, s0, α) = α * s * gammahazard(μ, cv2, s - s0)
	@register_symbolic γdivision(μ, cv2, s, s0, α)
	
	function partition_molecules(x, p)
		# Binomial partitioning of cell molecules x.
	    x_ = round(Int, x) # round to nearest Int to deal with floating point errors.
	    x1 = rand(Binomial(x_, p))
	    x2 = x_ - x1
	    return [x1, x2]
	end
	
	@register_symbolic partition_molecules(a, p)

	@parameters Cτ[1:2] Cs[1:2] Cs0[1:2] Cp[1:2] B[1:2] β μsize cv2size L 

	division = @interaction begin
	    @channel γdivision($μsize, $cv2size, $Cs[1], $Cs0[1], $α) / $C, 2*$C --> 2*$C
	    @sampler ExtrandeMethod($L; boundtype=:increasing) # The rate of division is increasing
		@variable $β = rand($Beta(100))
	    @variable $B = $partition_molecules($Cp[1], $β)
	    @transition (
	        ($s => $β*$Cs[1], $s0 => $β*$Cs[1], $p => $B[1]),
	        ($s => (1-$β)*$Cs[1], $s0 => (1-$β)*$Cs[1], $p => $B[2]))
	    @connections (
	        ($Cs[1] => $s, $Cp[1] => $p, $Cs0[1] => $s0,),
	        ())
	    @saveinstate ($C, $s) ($C, $p)
	    @saveoutstate ($C, $s) ($C, $p)
	    @name "division"
	end
end;

# ╔═╡ 099ce71c-6804-4c1a-91ed-9146387cf29a
md"""
We use the gamma hazard to define a rate function γdivision for cells to divide at size s and protein count p given initial size s0 and growth rate α. This rate is monotonically increasing and hence we set @sampler ExtrandeMethod($L; boundtype=:increasing).
"""

# ╔═╡ c3e142d8-9a28-4fea-84e4-34dccd37f380
md"""
## Transport
"""

# ╔═╡ 8105db17-6d04-416e-b682-ff7984ec32a0
md"""
Cell-to-cell communication is model by the following two transform interactions. The interaction ```pexport``` models the export of the signalling molecules into the environment. The interaction ```pimport``` models the import of the signalling molecules from the environment into a cell.
"""

# ╔═╡ 8fb49831-ae48-4f1d-9436-e481e45adab1
begin
	@parameters Ep rtransport
	pexport = @interaction begin
	    @channel $rtransport * $Cp[1], $C + $Env --> $C + $Env
	    @sampler ExtrandeMethod($L; boundtype=:increasing)
	    @connections (
	        ($Cs[1] => $s, $Cp[1] => $p, $Cs0[1] => $s0,),
	        ($Ep => $p,))
	    @transition (
	        ($p => $Cp[1] - 1, $s => $Cs[1], $s0 => $Cs0[1],),
	        ($p => $Ep +1 ), )
	end
	
	pimport = @interaction begin
	    @channel ($rtransport * $Ep), $C + $Env --> $C + $Env
	    @sampler GillespieMethod() 
	    @connections (
	        ($Cs[1] => $s, $Cp[1] => $p, $Cs0[1] => $s0,),
	        ($Ep => $p,))
	    @transition (
	        ($p => $Cp[1] + 1, $s => $Cs[1], $s0 => $Cs0[1]),
	        ($p => $Ep - 1, ))
	end
end;

# ╔═╡ 0b6e84ce-c153-45c8-b382-da4b80be4d17
md"""
## Population model.
"""

# ╔═╡ 8cd41d0c-aa0a-422c-85b6-8642cd492652
md"""
We combine the defied interactions and agent types into the AgentsModel.
"""

# ╔═╡ 8b100bdb-3626-447f-a08c-ab4e0b70fbae
population_model = AgentsModel([division, pexport, pimport], Dict(C => Cell, Env => Environment));

# ╔═╡ a24f9ead-8043-4bf9-9874-4b36c4be2c49
md"""
# Simulate
"""

# ╔═╡ 7667eb26-6353-4541-8ef5-ec30b34c636b
md"""
Set the initial conditions, parameters and simulate the model for 2000 time units.
"""

# ╔═╡ b4999299-1380-4b4a-a5bf-461401838ddc
function make_simulation_params(K_, popsize)
    # Initial population state. Starting with a single cell at age 0.
    init_cell = fill(C => (p => 1.0, s => 20.0, s0 => 20.0), popsize)
	# No autoinducer in the environment.
    init_env = [Env => (p => 0.0, ), ]
    init_pop=[init_cell ; init_env]

    Ω = 2 
    # Simulator timestep.
    Δt = 1.0
    tspan_pop = (0.0, 2000.0)

    sparams = SimulationParameters(
        [α => 1.0, kprod => Ω*1.0, μsize => 10.0, cv2size => 0.1,
		 rtransport => 0.05, K => Ω*K_, d => Ω*0.3, L => 0.1], 
        tspan_pop, 
        Δt; 
        snapshot=[PopulationSnapshot(C), StateSnapshot(Env, p), StateSnapshot(C, p)])

    return sparams, init_pop
end;


# ╔═╡ 9de6ff11-3ef7-4c76-b6a2-969e0b7eea5a
begin
	sparams5, init_pop5 = make_simulation_params(0.4, 5)
	sparams20, init_pop20 = make_simulation_params(0.4, 20)
	
	res_pop5 = simulate(population_model, init_pop5, sparams5; trace_agents=true, save_interactions=true) 
	res_pop20 = simulate(population_model, init_pop20, sparams20; trace_agents=true, save_interactions=true)
end

# ╔═╡ 9586b926-0309-41d2-a96e-97ba104104c7
md"""
# Plotting
"""

# ╔═╡ 25ae1ba3-600d-4f12-b6a5-142a54a2c275
md"""
First let's define some helper functions.
"""

# ╔═╡ 30854098-0274-4c85-a21e-afe7950a003f
begin
	# Colorscheme
	colors = ColorSchemes.Hiroshige.colors
	transp = 0.8
	htransp = 0.05
	stairstransp = 0.9
	barstransp = 0.3

	function plot_hist!(ax, values; bins, color, label)
	    hist_ = normalize(fit(Histogram, values, bins); mode=:pdf)
	    stairs!(ax, collect(midpoints(hist_.edges[1])), hist_.weights; color=(color, stairstransp), step=:center, label=label)
	    barplot!(ax, collect(midpoints(hist_.edges[1])), hist_.weights; 
	        color=(color, barstransp), strokecolor=(color, transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)
	end
	
	function plot_traj!(ax, res; color)
	    for cell in collect(values(res.final_pop[C]))[1:1]
	        lin = AgentBasedModeling.lineage(res, cell)
	        traj_t = []
	        traj_u = []
	        for cell in lin
	            btime = cell.btime
	            dtime = !isnothing(cell.dtime) ? cell.dtime : cell.simulation.t[end]
	            isinf(dtime) && break
	            tspan = btime:1.0:dtime
	            push!(traj_t, collect(tspan)...)
	            push!(traj_u, cell.simulation(tspan; idxs=p).u...)
	        end
	        lines!(ax, traj_t, traj_u; color=color[1], linewidth=1.0)
	    end
	    
	    lines!(ax, res.snapshot[:Envp].t, getindex.(res.snapshot[:Envp].u, 1); color=color[2], linewidth=1) 
	end
end;

# ╔═╡ e9ece2ae-5994-4d3e-b521-be87b8198e14
md"""
Plot autoinducer counts in cells and environment.
"""

# ╔═╡ 88906c28-5735-4e49-bf29-eea54b8391b0
begin
	# Figure layout
	pt_cm = 2.83465
	fig = Figure(size=(170, 50*2) .* pt_cm; fontsize=8, pt_per_unit=1)
	ax_protein5 = Axis(fig[2,1]; xlabel="Autoinducer counts", ylabel="Probability density")
	ax_protein20 = Axis(fig[1,1]; xlabel="Autoinducer counts", ylabel="Probability density")
	ax_traj5 = Axis(fig[2,2]; xlabel="Time", ylabel="p")
	ax_traj20 = Axis(fig[1,2]; xlabel="Time", ylabel="p")
	
	plot_hist!(ax_protein5, reduce(vcat, res_pop5.snapshot[:Cp].u); color=colors[3], bins=0:1:50, label="Cell snapshot distribution")
	plot_hist!(ax_protein5, reduce(vcat, res_pop5.snapshot[:Envp].u); color=colors[8], bins=0:1:50, label="Environment snapshot distribution")
	xlims!(ax_protein5, (0, 40))
	ylims!(ax_protein5, low=0)
	#axislegend(ax_protein)
	
	plot_hist!(ax_protein20, reduce(vcat, res_pop20.snapshot[:Cp].u); color=colors[3], bins=0:1:50, label="Cell snapshot distribution")
	plot_hist!(ax_protein20, reduce(vcat, res_pop20.snapshot[:Envp].u); color=colors[8], bins=0:1:50, label="Environment snapshot distribution")
	xlims!(ax_protein20, (0, 40))
	ylims!(ax_protein20, low=0)
	#axislegend(ax_protein)
	
	plot_traj!(ax_traj5, res_pop5; color=(colors[3], colors[8]))
	xlims!(ax_traj5, (0, 1000))
	ylims!(ax_traj5, (0, 50))
	plot_traj!(ax_traj20, res_pop20; color=(colors[3], colors[8]))
	xlims!(ax_traj20, (0, 1000))
	ylims!(ax_traj20, (0, 50))
	
	hidedecorations!.(fig.content[1:4], ticklabels = false, ticks = false, label = false)
	hidespines!.(fig.content[1:4], :t, :r)
end;

# ╔═╡ bf357515-2b32-4e8a-a1d5-22ac9d7c80f5
fig

# ╔═╡ 5eadbf20-f2a6-410d-977a-914a95ffccbd
md"""
# Lineage plot
"""

# ╔═╡ c962fa91-47ea-429e-8604-770eca179053
md"""
Detrended layout helper function.
"""

# ╔═╡ 8aef32f2-0a41-4d44-b73a-c4e381998383
begin
	function mylayout(g; offset=0)
	    meta = [g.vertex_properties[uid][2] for uid in labels(g)]
	    buch = NetworkLayout.buchheim(adjacency_matrix(g))
	
	    xs = getindex.(meta, 2)
	    ys = getindex.(buch, 1) .- offset
	
	    y1 = ys[end]
	    x1 = xs[end] 
	    m = y1 / x1
	
	    nbins = 50
	
	    range_ = range(0.0, stop=y1, length=nbins)
	    bins = collect(zip(range_, range_[2:end]))
	
	    binsx = [[] for i = 1:length(bins)] 
	    detrend = zip(xs, ys .- m .* xs)
	
	    for (x, y) in detrend 
	        idx = findfirst(b -> b[1] < x < b[2], bins)
	        isnothing(idx) && continue
	        push!(binsx[idx], y)
	    end
	
	    mbins = [!isempty(x) ? mean(x) : 0.0 for x in binsx]
	    itp = linear_interpolation(midpoints(range_), mbins; extrapolation_bc=Line())
	    
	    return Point.([(x, y - itp(x)) for (x,y) in detrend])
	end
end;

# ╔═╡ 6e2b0eef-2e0b-4bb2-bdce-289f523effa7
begin
	# Construct the graph.
	agents = MetaGraph(
	    Graphs.SimpleDiGraph();
	    label_type=UInt,
	    vertex_data_type=Any,
	    edge_data_type=Any,
	    weight_function=identity);
	
	for interaction in res_pop5.interactions
	    add_vertex!(agents, interaction[2].uid, (interaction[2].pitx.itxdef.name, interaction[1]))
	
	    for (sym, uid) in interaction[2].substrates
	        isequal(sym, Env) && continue
	        agent = res_pop5.agents[sym][uid]
	        !in(agent.dinteraction, keys(agents.vertex_properties)) && continue
	
	        ts = range(agent.btime, stop=agent.dtime, length=10) 
	
	        (indegree(agents, agents.vertex_properties[agent.dinteraction][1]) >= 1) &&  begin
	            add_vertex!(agents, uid, (sym, agent.dtime)) 
	            agents[agent.dinteraction, uid] = [agent.sym, agent.dtime - agent.btime, agent.uid, mean(agent.simulation(ts; idxs=p))]
	            continue
	        end
	        agents[agent.binteraction, agent.dinteraction] = [agent.sym, agent.dtime - agent.btime, agent.uid, mean(agent.simulation(ts; idxs=p))]
	    end
	end
	
	#Plot results
	fig_graph = Figure(size=(200, 50) .* pt_cm; fontsize=12, pt_per_unit=1)
	ax = Axis(fig_graph[1,2], xlabel="time")
	
	hidedecorations!.(fig_graph.content, ticks = false, label = false, ticklabels=false)
	hideydecorations!.(fig_graph.content)
	hidespines!.(fig_graph.content, :t, :r, :l) 
	
	color_ = Dict("infection" => colors[1], "division" => :black, "lysis" => colors[3], "uptake" =>  RGB(8.0/256, 217.0/256, 41.0/256))
	
	cnx_comps = connected_components(agents)
	
	for (i, comp) in enumerate(cnx_comps)
	    subg = induced_subgraph(agents, comp)[1]
	    lay_ = mylayout(subg)
	
	    meta = [subg.edge_data[uid][4] for uid in edge_labels(subg)]
	    for edge in edges(subg)
	        src_ = edge.src 
	        dst_ = edge.dst 
	        meta = subg.edge_data[(subg.vertex_labels[src_], subg.vertex_labels[dst_])][4]
	        lines!(ax, [lay_[src_], lay_[dst_]]; color=meta, colorrange=(0.0, 10.0), colormap=range(colors[4], stop=colors[1], length=10), linewidth=1.0)
	    end
	end
	
	Colorbar(fig_graph[1, 1], limits = (10, 60), colormap=range(colors[4], stop=colors[1], length=10), flipaxis = false, label="phage counts")
	resize_to_layout!(fig_graph)
	
	melems = [MarkerElement(color = c, marker = :circle) for c in values(color_)]
	
	xlims!(ax, high=4.0)
	xlims!(ax, low=0.0, high=500)
end;

# ╔═╡ 3453f3ab-cc95-4f65-8f71-7dd4280c84e0
fig_graph

# ╔═╡ Cell order:
# ╟─ea9c5a6d-7225-43b7-b3f9-562f2185f5b0
# ╠═fd7efc3c-ff22-11ef-34c6-dfce6f883a42
# ╟─6b3dab0d-b5c5-4632-a7f2-43fa0a07b01f
# ╟─2cdf4d9f-0374-4bb6-bf7f-f34361d1a6b7
# ╠═1279c064-b5ea-4ea7-8f70-097a4ce97490
# ╟─5b42ff0f-ef47-487b-aff5-188b06f13c96
# ╟─e51ee76f-c8b9-4884-9cc9-7928cd4e0f2f
# ╠═978707eb-3ae3-4cdf-b5a0-28b16e2cbc13
# ╟─a8cff812-98ae-42cc-b1de-0130b5326f12
# ╟─b549d1a5-4e90-417a-9890-7ad682f7f6af
# ╟─03dba07e-a5b9-45cc-9f6e-1572a3bc5fb0
# ╠═0077a4fa-340c-411c-b95e-56a03bb3fb0b
# ╠═099ce71c-6804-4c1a-91ed-9146387cf29a
# ╠═41f432e8-9049-4b75-83a3-0bbe464f1c52
# ╟─c3e142d8-9a28-4fea-84e4-34dccd37f380
# ╟─8105db17-6d04-416e-b682-ff7984ec32a0
# ╠═8fb49831-ae48-4f1d-9436-e481e45adab1
# ╟─0b6e84ce-c153-45c8-b382-da4b80be4d17
# ╟─8cd41d0c-aa0a-422c-85b6-8642cd492652
# ╠═8b100bdb-3626-447f-a08c-ab4e0b70fbae
# ╟─a24f9ead-8043-4bf9-9874-4b36c4be2c49
# ╠═7667eb26-6353-4541-8ef5-ec30b34c636b
# ╠═b4999299-1380-4b4a-a5bf-461401838ddc
# ╠═9de6ff11-3ef7-4c76-b6a2-969e0b7eea5a
# ╟─9586b926-0309-41d2-a96e-97ba104104c7
# ╟─25ae1ba3-600d-4f12-b6a5-142a54a2c275
# ╠═30854098-0274-4c85-a21e-afe7950a003f
# ╟─e9ece2ae-5994-4d3e-b521-be87b8198e14
# ╠═88906c28-5735-4e49-bf29-eea54b8391b0
# ╠═bf357515-2b32-4e8a-a1d5-22ac9d7c80f5
# ╟─5eadbf20-f2a6-410d-977a-914a95ffccbd
# ╟─c962fa91-47ea-429e-8604-770eca179053
# ╠═8aef32f2-0a41-4d44-b73a-c4e381998383
# ╠═6e2b0eef-2e0b-4bb2-bdce-289f523effa7
# ╠═3453f3ab-cc95-4f65-8f71-7dd4280c84e0
