### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ c2bc53d4-fe77-11ef-2f79-7b2519b0b80d
# ╠═╡ show_logs = false
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
		Pkg.PackageSpec(name="OrdinaryDiffEq"),
        Pkg.PackageSpec(name="Catalyst"),
		Pkg.PackageSpec(name="Distributions"),
		Pkg.PackageSpec(name="CairoMakie"),
		Pkg.PackageSpec(name="GraphMakie"),	
		Pkg.PackageSpec(name="ColorSchemes"),
		Pkg.PackageSpec(name="Colors"),
		Pkg.PackageSpec(name="StatsBase"),
		Pkg.PackageSpec(name="Integrals"),
		Pkg.PackageSpec(name="Interpolations"),
		Pkg.PackageSpec(name="Graphs"),
		Pkg.PackageSpec(name="MetaGraphsNext"),
		Pkg.PackageSpec(name="NetworkLayout"),
		Pkg.PackageSpec(url="https://github.com/pihop/AgentBasedModeling.jl")
    ])
	using OrdinaryDiffEq
	using AgentBasedModeling
	using Catalyst
	using Distributions
	using LinearAlgebra
	using CairoMakie
	using GraphMakie
	using StatsBase
	using Integrals
	using Graphs
	using MetaGraphsNext
	using NetworkLayout
	using Interpolations
	using ColorSchemes
	using Colors
end;

# ╔═╡ 0d78e430-a18f-4795-8772-261a2212936d
md"""
# Cell phage example
"""

# ╔═╡ 3bb91f15-8249-433c-945d-16ea62f644a7
md"""
# Cell state dynamics
"""

# ╔═╡ e2a4ee0e-7020-4835-a575-dab4dccebed5
md"""
The cell state dynamics descibe the growth of the cell size, aging, replication of phages and consumption of nutrients.
"""

# ╔═╡ 5e2a23c5-b35b-4101-a70a-398cf95525ef
begin
	@independent_variables t 
	@abm_variables s(t) τ(t) p(t) N(t) InfCount(t)
	@parameters s0 α K kprod
	D = Differential(t)
		
	cell_dynamics = @reaction_network begin
        @variables s(t) τ(t)
		@species p(t) N(t) InfCount(t)
	    @equations begin
	        D(s) ~ hill(N, α, K, 1.0)*s
	        D(τ) ~ 1.0
	    end
	    kprod, p + N --> 2*p
	    hill(N, α, K, 1), N --> 0
	end
	
	Cell = AgentDynamics(cell_dynamics, (s0, ))
end;

# ╔═╡ 62f11bb4-8bd1-4bee-962a-d93494e841b9
md"""
# Environment dynamics
"""

# ╔═╡ 856a9d9a-d1d0-4231-926e-90f8ac7625a8
md"""
The environment agent's state keeps track of the number of available nutrients N and phages p in the environment.
"""

# ╔═╡ fac21459-d60d-41d1-bbc7-0b13c889b0cd
Environment = AgentDynamics(EmptyTraitProblem(), (N, p));

# ╔═╡ be932fa8-06e9-4e8e-bbc1-d81390dcc7ca
md"""
# Interactions
"""

# ╔═╡ 2b76895b-6f97-4905-9035-2bb3a0738cc7
md"""
## Cell division
"""

# ╔═╡ abbbb689-484f-4ce3-af4d-af3b8bda0ec4
md"""
Hazard function for the gamma distribution. Gammahaz function parametrised by the mean μ and (cv)^2 of the distribution sets the hazard value of the right tail to the theoretical limit 1/(μ*(cv)^2).
"""

# ╔═╡ 7641cde7-8496-49a4-a90f-040af64b5f01
md"""
We use the gamma hazard to define a rate function γdivision for cells to divide at size s and protein count p given initial size s0 and growth rate α. This rate is monotonically increasing and hence we set `@sampler ExtrandeMethod($L; boundtype=:increasing)`.
"""

# ╔═╡ 71ba5cb9-17e8-4e2f-bfe0-0dcabd78e390
begin
	hazard(d::Gamma, x) = exp(logpdf(d, x) - logccdf(d, x))
	function gammahazard(μ, cv2, x) 
	    g = Gamma(1/cv2, μ*cv2)
	    x > μ + 4*std(g) && return 1/(μ*cv2)
	    return hazard(g, x)
	end
	
	γdivision(μ, cv, s, s0, α, p) = ( p == 0 ? 1. : 0. ) * α * s * gammahazard(μ, cv, s - s0)
	@register_symbolic γdivision(μ, cv, s, s0, α, p)
end;

# ╔═╡ 4ee00772-6634-4499-a165-a6c3e5109c02
begin
	@abm_variables C(t)
	@parameters B[1:2, 1:2]
	@parameters Cτ Cs Cs0 Cp CN β
	@parameters μsize cvsize L
	
	function partition_molecules(x, y, β)
	    x_ = round(Int, x)
	    x1 = rand(Binomial(x_, β))
	    x2 = x_ - x1
	
	    y_ = round(Int, y)
	    y1 = rand(Binomial(y_, β))
	    y2 = y_ - y1
	    return [x1 x2; y1 y2]
	end
	
	@register_symbolic partition_molecules(a, b, β)
	
	division = @interaction begin
	    @channel γdivision($μsize, $cvsize, $Cs, $Cs0, $α, $Cp), $C --> 2*$C
	    @sampler ExtrandeMethod($L; boundtype=:increasing)
		@variable $β = rand($Beta(100))
	    @variable $B = $partition_molecules($Cp, $CN, $β)
	    @transition (
	        ($τ => 0.0, $s => $β*$Cs, $s0 => $β*$Cs, $p => $B[1,1], $N => $B[2,1], $InfCount => 0.0),
	        ($τ => 0.0, $s => (1-$β)*$Cs, $s0 => (1-$β)*$Cs, $p => $B[1,2], $N => $B[2,2], $InfCount => 0.0))
	    @connections (
	        ($Cτ => $τ, $Cs => $s, $Cp => $p, $Cs0 => $s0, $CN => $N),)
	    @saveinstate ($C, $s) ($C, $p)
	    @saveoutstate ($C, $s) ($C, $p)
	    @name "division"
	end
end;

# ╔═╡ 268697d9-4464-4bcb-acaa-ed60431d757f
md"""
## Infection
"""

# ╔═╡ 2602b90c-7478-4e1d-a5e3-c7c62db94a67
md"""
The infection interaction is modelled as an interaction of the cell with the environemnts that increases the count of phages in the cell by 1 and decreases the phages in the environment by 1. We assume the infections happen with a constant rate `kinf` following the law of mass action and thus Gillespie algorithm can be used for the sampling of interaction times.
"""

# ╔═╡ 4d62aa7c-7c9c-46b8-8168-bcf42fab1ed0
begin
	@abm_variables Env(t)
	@parameters CInfCount Ep EN
	@parameters kinf

	infection = @interaction begin
	    @channel $kinf*$Ep, $C + $Env --> $C + $Env
	    @sampler GillespieMethod()
	    @connections (
	        ($Cτ => $τ, $Cs => $s, $Cp => $p, $Cs0 => $s0, $CN => $N, $CInfCount => $InfCount),
	        ($Ep => $p, $EN => $N))
	    @transition (
	        ($p => $Cp + 1, 
	         $τ => ifelse($Cp > 0, $Cτ, 0.0), 
	         $s => $Cs, 
	         $s0 => $Cs0, 
	         $N => $CN, 
	         $InfCount => $CInfCount + 1),
	        ($p => $Ep - 1, $N => $EN))
	    @name "infection"
	end
end;

# ╔═╡ c606780b-d26f-4506-8733-cfa05d84a070
md"""
## Lysis
"""

# ╔═╡ 78b6f6c1-ecc7-402d-a4df-443b88fde9ca
md"""
Lysis of infected cells happens with a rate dependent on the time since infection τ. The time to lysis distribution is Gamma distributed with mean μlysis and coefficient of variation square `cvlysis`.
"""

# ╔═╡ d79fd199-1ea5-4b4b-981d-228f77266db0
begin
	@parameters μlysis cvlysis
	γlysis(μ, cv2, τ, p) = ( p > 0 ? 1. : 0. )*gammahazard(μ, cv2, τ)
	@register_symbolic γlysis(μ, cv2, τ, p)
	
	lysis = @interaction begin
	    @channel γlysis(μlysis, cvlysis, $Cτ, $Cp), $C + $Env --> $Env
	    @sampler ExtrandeMethod($L; boundtype=:increasing)
	    @connections (
	        ($Cτ => $τ, $Cs => $s, $Cp => $p, $Cs0 => $s0, $CN => $N),
	        ($Ep => $p, $EN => $N))
	    @transition (
	        ($p => $Ep + $Cp, $N => $EN), )
		@saveinstate ($C, $p) ($C, $InfCount)
	    @name "lysis"
	end
end;

# ╔═╡ 836d8a9d-a469-41ef-9985-14027df97795
md"""
## Uptake
"""

# ╔═╡ 5f24eeba-9d8b-4813-b33a-c930dab9417a
md"""
Cells take up nutrients from the environment at a constant rate `kuptake` following the law of mass action.
"""

# ╔═╡ 0ce6dcb1-4cae-46e2-86ec-779addfc29ac
begin
	@parameters kuptake
	uptake = @interaction begin
	    @channel $kuptake*$EN, $C + $Env --> $C + $Env
	    @sampler GillespieMethod()
	    @connections (
	        ($Cτ => $τ, $Cs => $s, $Cp => $p, $Cs0 => $s0, $CN => $N, $CInfCount => $InfCount),
	        ($Ep => $p, $EN => $N,))
	    @transition (
	        ($p => $Cp, $τ => $Cτ, $s => $Cs, $s0 => $Cs0, $N => $CN + 1, $InfCount => $CInfCount),
	        ($p => $Ep, $N => $EN - 1, ))
		@name "uptake"
	end
end;

# ╔═╡ 183e596d-c979-4be0-8143-e792f2228015
md"""
# Population model
"""

# ╔═╡ 9c6353c0-d812-4b3e-837d-9f76cf7c3702
md"""
We combine the defied interactions and agent types into the `AgentsModel`.
"""

# ╔═╡ 006fca5c-5ce7-4a77-a8b3-d3cef0c023e0
population_model = AgentsModel([division, infection, lysis, uptake], Dict(C => Cell, Env => Environment));

# ╔═╡ e0adfa6c-d553-4a22-8623-1927dfed574a
md"""
And define the initial populations and simulation parameters.
"""

# ╔═╡ 8fc6936a-d837-4331-aafc-3b59f7bc922d
begin
	# Initial population of single cell and environment.
	init_cell = fill(C => (p => 0.0, τ => 0.0, s => 0.50, s0 => 0.50, N => 100.0, InfCount => 0.0), 1)
	init_env = [Env => (p => 10.0, N => 100.0), ]
	init_pop=[init_cell ; init_env]
	tspan_pop = (0.0, 20.0) # Simulation tspan
	Δt = 0.1 # Simulation timestep
	
	simulation_params = SimulationParameters(
	    [α => 7.0, kprod => 10.0, μlysis => 0.8, cvlysis => 0.1, 
		 L => 0.01, kinf => 0.1, μsize => 1.0, cvsize => 0.1, K => 5, kuptake => 0.1],
	    tspan_pop, 
	    Δt; 
	    snapshot=[PopulationSnapshot(C), 
	              StateSnapshot(C, p), 
	              StateSnapshot(Env, p), 
	              StateSnapshot(Env, N), 
	              StateSnapshot(C, s), 
	              StateSnapshot(C, InfCount), 
	              StateSnapshot(C, τ)], 
	    maxpop=1000)
end;

# ╔═╡ 5cfc9c3f-0065-4fc2-a1cd-609f12d877f1
md"""
# Simulate model 
To keep the simulation time for the notebook low we simulate 50 trajectories. To replicated distributions from the paper change to 1500.
"""

# ╔═╡ 8e14c51f-353a-4557-ac9e-72111c6b8e96
# ╠═╡ show_logs = false
ressim = [simulate(population_model, init_pop, simulation_params; trace_agents=true, save_interactions=true) for i in 1:50];

# ╔═╡ 2f5d7c98-8b13-4136-b7b5-432c6ee20dd8
md"""
# Plotting
"""

# ╔═╡ c05c85d4-da79-444d-9cdb-9427fbc351d0
begin
	CairoMakie.activate!()
	colors = ColorSchemes.Hiroshige.colors
	transp = 0.8
	htransp = 0.05
	stairstransp = 0.9
	barstransp = 0.3

	pt_cm = 2.83465
	fig = Figure(size=(170, 50*2) .* pt_cm; fontsize=8, pt_per_unit=1)
	ax_cells = Axis(fig[1,1]; xlabel="Time", ylabel="Cells")
	ax_phages = Axis(fig[1,2]; xlabel="Time", ylabel="Phages")
	ax_burst = Axis(fig[2,1]; xlabel="Phage burst size", ylabel="Density")
	ax_ext = Axis(fig[2,2]; xlabel="Extinction time", ylabel="Density")

	function plot_hist!(ax, values; bins, color, label="", weight = 1.0, plotbars = true)
	    hist_ = normalize(fit(Histogram, values, bins); mode=:pdf)
	    stairs!(ax, collect(midpoints(hist_.edges[1])), hist_.weights .* weight; color=(color, stairstransp), step=:center, label=label)
	    plotbars && barplot!(ax, collect(midpoints(hist_.edges[1])), hist_.weights .* weight; 
	        color=(color, barstransp), strokecolor=(color, transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)
	end

end;

# ╔═╡ 1ff7ace4-0b3f-4c89-b3be-17fd9a134548
begin
	ext_times = round.([res.snapshot[:C].t[findfirst(x -> x == 0, res.snapshot[:C].u)] for res in ressim], digits=3)
	burst_sizes = [getindex.(res.instates[:Cp_lysis].u, 1) for res in ressim]
	peaks = reduce(vcat, [maximum(res.snapshot[:C].u) for res in ressim])
	classes = ext_times .< 2.5 
end;

# ╔═╡ e9636721-71d2-43d1-ba1f-5740bae989bb
begin
	for res in rand(ressim[classes .== 0], 5)
	    col = colors[1]
	    lines!(ax_cells, res.snapshot[:C].t, res.snapshot[:C].u; color=(col, transp))
	    lines!(ax_phages, res.snapshot[:Envp].t, getindex.(res.snapshot[:Envp].u, 1); color=(col, transp))
	end
	
	for res in rand(ressim[classes .== 1], 5)
	    col = colors[9]
	    lines!(ax_cells, res.snapshot[:C].t, res.snapshot[:C].u; color=(col, transp))
	    lines!(ax_phages, res.snapshot[:Envp].t, getindex.(res.snapshot[:Envp].u, 1); color=(col, transp))
	end

	plot_hist!(ax_burst, reduce(vcat, burst_sizes[classes]); color=colors[9], bins=0:2:140)
	plot_hist!(ax_burst, reduce(vcat, burst_sizes[.!classes]); color=colors[1], bins=0:2:140, weight = length(ext_times[classes]) / length(classes))
		
	plot_hist!(ax_ext, ext_times[.!classes]; color=colors[1], bins=0.0:0.2:10.0, weight = length(ext_times[.!classes]) / length(classes))
	plot_hist!(ax_ext, ext_times[classes]; color=colors[9], bins=0.0:0.2:10.0, weight = length(ext_times[classes]) / length(classes))

	xlims!.([ax_cells, ax_phages, ax_ext], Ref((0, 7.5)))
	xlims!(ax_ext, (0, 10))
	xlims!(ax_burst, (0, 140))
	ylims!.(fig.content, low=0)
end;

# ╔═╡ b7196a8f-f42a-429b-85f8-5b3cd802caa0
fig

# ╔═╡ 972d9160-26df-43bf-bd3e-448623a724e4
md"""
## Interaction graph

Making use of the `construct_intreaction_graph` function provided by the package.
"""

# ╔═╡ 95eb5b09-8532-4877-a1a8-6940cbadff3b
begin
	itx_graph = construct_interaction_graph(ressim[1]; agent_filter=[Env,])
end

# ╔═╡ 6c049394-204a-4011-ab6c-7cc294fa54e2
begin
	function edge_data(graph, edge)
	    vlabels = graph.vertex_labels
	    agent = graph.edge_data[vlabels[edge.src], vlabels[edge.dst]]
	    ts = range(agent.btime, stop=agent.dtime, length=10) 
	    return agent.simulation(ts; idxs=p).u
	end
	
	function make_layout(graph)
	    meta = [graph.vertex_properties[uid][2] for uid in labels(graph)]
	    buch = NetworkLayout.buchheim(adjacency_matrix(graph))
	
	    xs = getindex.(meta, 2)
	    ys = getindex.(buch, 1)
	    return Point.(zip(xs, ys))
	end
	
	function vertex_colors(g; color)
	    meta = [g.vertex_properties[uid][2] for uid in labels(g)]
	    ys = [color[m] for m in getindex.(meta, 1)]
	    return ys
	end
	
	function vertex_data(g)
	    meta = [g.vertex_properties[uid][2] for uid in labels(g)]
	    return getindex.(meta, 1) 
	end
	
	fig_graph = Figure(size=(700, 200))
	ax = Axis(fig_graph[1,2], xlabel="time")
	
	hidedecorations!.(fig_graph.content, ticks = false, label = false, ticklabels=false)
	hideydecorations!.(fig_graph.content)
	hidespines!.(fig_graph.content, :t, :r, :l) 
	
	color_ = Dict("infection" => colors[1], "division" => :black, "lysis" => colors[3], "uptake" =>  RGB(8.0/256, 217.0/256, 41.0/256))
	
	lay_ = make_layout(itx_graph)
	for edge in edges(itx_graph)
	    data_ = edge_data(itx_graph, edge)
	    src_ = edge.src 
	    dst_ = edge.dst 
	    lines!(ax, [lay_[src_], lay_[dst_]]; color=data_, colorrange=(0.0, 10.0), colormap=range(colors[7], stop=colors[1], length=10), linewidth=2)
	end
	
	graphplot!(ax, itx_graph, curves=false; 
	    layout=make_layout, 
	    node_color=vertex_colors(itx_graph; color=color_),
	    edge_color=(colors[1], 0.0),
	    nlabels_align=(:center, :center),
	    nlabels_offset=Point2f(0.0, 0.4),
	    elabels_offset=Point2f(0.0, 0.1),
	    edge_width=3,
	    node_size=6
	   )
	
	Colorbar(fig_graph[1, 1], limits = (10, 60), colormap=range(colors[7], stop=colors[1], length=10), flipaxis = false, label="phage counts")
	resize_to_layout!(fig_graph)
	
	melems = [MarkerElement(color = c, marker = :circle) for c in values(color_)]
	
	Legend(fig_graph[1,3], melems, collect(keys(color_)); framevisible=false)
	
	#scatter!(ax, lay_; color=:black, markersize=5)
	xlims!(ax, high=4.0)
	xlims!(ax, low=0.0)
end

# ╔═╡ 1850bcbb-c6c4-4da9-b44f-17fb93480ccb
fig_graph

# ╔═╡ Cell order:
# ╟─0d78e430-a18f-4795-8772-261a2212936d
# ╠═c2bc53d4-fe77-11ef-2f79-7b2519b0b80d
# ╟─3bb91f15-8249-433c-945d-16ea62f644a7
# ╟─e2a4ee0e-7020-4835-a575-dab4dccebed5
# ╠═5e2a23c5-b35b-4101-a70a-398cf95525ef
# ╟─62f11bb4-8bd1-4bee-962a-d93494e841b9
# ╟─856a9d9a-d1d0-4231-926e-90f8ac7625a8
# ╠═fac21459-d60d-41d1-bbc7-0b13c889b0cd
# ╟─be932fa8-06e9-4e8e-bbc1-d81390dcc7ca
# ╟─2b76895b-6f97-4905-9035-2bb3a0738cc7
# ╟─abbbb689-484f-4ce3-af4d-af3b8bda0ec4
# ╟─7641cde7-8496-49a4-a90f-040af64b5f01
# ╠═71ba5cb9-17e8-4e2f-bfe0-0dcabd78e390
# ╠═4ee00772-6634-4499-a165-a6c3e5109c02
# ╟─268697d9-4464-4bcb-acaa-ed60431d757f
# ╟─2602b90c-7478-4e1d-a5e3-c7c62db94a67
# ╠═4d62aa7c-7c9c-46b8-8168-bcf42fab1ed0
# ╟─c606780b-d26f-4506-8733-cfa05d84a070
# ╟─78b6f6c1-ecc7-402d-a4df-443b88fde9ca
# ╠═d79fd199-1ea5-4b4b-981d-228f77266db0
# ╟─836d8a9d-a469-41ef-9985-14027df97795
# ╟─5f24eeba-9d8b-4813-b33a-c930dab9417a
# ╠═0ce6dcb1-4cae-46e2-86ec-779addfc29ac
# ╟─183e596d-c979-4be0-8143-e792f2228015
# ╟─9c6353c0-d812-4b3e-837d-9f76cf7c3702
# ╠═006fca5c-5ce7-4a77-a8b3-d3cef0c023e0
# ╟─e0adfa6c-d553-4a22-8623-1927dfed574a
# ╠═8fc6936a-d837-4331-aafc-3b59f7bc922d
# ╟─5cfc9c3f-0065-4fc2-a1cd-609f12d877f1
# ╠═8e14c51f-353a-4557-ac9e-72111c6b8e96
# ╟─2f5d7c98-8b13-4136-b7b5-432c6ee20dd8
# ╠═c05c85d4-da79-444d-9cdb-9427fbc351d0
# ╠═1ff7ace4-0b3f-4c89-b3be-17fd9a134548
# ╠═e9636721-71d2-43d1-ba1f-5740bae989bb
# ╠═b7196a8f-f42a-429b-85f8-5b3cd802caa0
# ╟─972d9160-26df-43bf-bd3e-448623a724e4
# ╠═95eb5b09-8532-4877-a1a8-6940cbadff3b
# ╠═6c049394-204a-4011-ab6c-7cc294fa54e2
# ╠═1850bcbb-c6c4-4da9-b44f-17fb93480ccb
