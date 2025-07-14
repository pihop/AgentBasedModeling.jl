### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 355d1e6c-91f6-11ef-2f45-19eb05b5141f
# ╠═╡ show_logs = false
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
		Pkg.PackageSpec(name="OrdinaryDiffEq"),
        Pkg.PackageSpec(name="Catalyst"),
		Pkg.PackageSpec(name="Distributions"),
		Pkg.PackageSpec(name="JumpProcesses"),
		Pkg.PackageSpec(name="CairoMakie"),
		Pkg.PackageSpec(name="ColorSchemes"),
		Pkg.PackageSpec(name="StatsBase"),
		Pkg.PackageSpec(name="Integrals"),
		Pkg.PackageSpec(name="Interpolations"),
		Pkg.PackageSpec(name="Graphs"),
		Pkg.PackageSpec(name="MetaGraphsNext"),
		Pkg.PackageSpec(name="NetworkLayout"),
		Pkg.PackageSpec(url="https://github.com/pihop/AgentBasedModeling.jl")
    ])
	using OrdinaryDiffEq
	using JumpProcesses
	using LinearAlgebra
	using CairoMakie
	using StatsBase
	using Integrals
	using Graphs
	using MetaGraphsNext
	using NetworkLayout
	using Interpolations
	using ColorSchemes
end;

# ╔═╡ 35c0a480-6582-4531-ae36-1135665ee564
begin
	using Catalyst, Distributions, AgentBasedModeling
	@variables t 
	@abm_variables Δ(t) s(t) p(t)
	@parameters α kprod b 
	D = Differential(t)
	
	@register_symbolic Distributions.Geometric(a)
	m = rand(Distributions.Geometric(1/(1 + b*s)))

	CellDynamics = @reaction_network begin
		@species p(t) Δ(t) s(t)
		@equations begin
			D(s) ~ $α*s 
	 		D(Δ) ~ $α*s 
	 		D(p) ~ 0.0
		end
    	kprod, 0 --> $m*p 
	end
end;

# ╔═╡ 4e3e20d3-d152-4c26-a1f8-a0f4d11cf025
md"""
# AgentBasedModeling.jl demonstration -- Stochastic gene expression in growing and dividing cells.
"""

# ╔═╡ 696a96c3-3499-41a7-a045-ec5d55806319
md"""
The pupose of this notebook is to demostrate the usage of AgentBasedModeling.jl and recreate the size control model from the paper.
"""

# ╔═╡ 93633661-a134-4f9d-89f3-497c38283181
md"""
We consider bursty protein production --- cells produce protein in geometrically distributed burst. The burst size depends on a parameter b and cell size s.

The burst happen with a rate kprod.

Let us set up the geometric distribution and the reaction network corresponding to the interal gene expression dynamics of cells.
"""

# ╔═╡ 019b50c4-dc51-45eb-bc28-770b8d5c2696
md"""
# Step 1
"""

# ╔═╡ d6827f12-e43b-4f19-bfa0-c8cc5d040fa4
md"""
The cells are also assumed to grow exponentially in size with a rate α. The cell age increases linearly with time. To model that we define an ODE system.
"""

# ╔═╡ 1759ae4c-60ec-493d-a2d4-06b7cc83a8f2
md"""
AgentBasedModeling exports the structure AgentDynamics which we use to collect the discrete and continous interal dynamics of the cells.
"""

# ╔═╡ fee337cd-3f0c-4ccd-a023-7aa4e717c678
Cell = AgentDynamics(CellDynamics, ());

# ╔═╡ 7fcf8886-a1f9-419f-aef7-18ab5a39ce52
md"""
Next we set up the functions defining cell division rate. In particular, let us consider the adder rule for cell division where the rate of cell divison depends on the size added since birth. The distributions of added size is given by a Gamma distribution with mean μ and cv2 (coefficient of variation squared).
"""

# ╔═╡ 6f00bd54-ffd2-4f41-a533-a35727ba590c
md"""
# Step 2
"""

# ╔═╡ a5ff7927-087b-43bb-80a9-d9d13c3fe2c5
begin
	γsize(x; μ, cv2) = exp(
    	logpdf(Gamma(1/cv2, μ*cv2), x) - logccdf(Gamma(1/μ, μ*cv2), x))

	# Adder rule for cell division.
	γdivision(s, Δ, α; μ = 1.0, cv2 = 0.2) = α * s * γsize(Δ; μ=μ, cv2=cv2)
	@register_symbolic γdivision(s, Δ, α)
end;

# ╔═╡ 7883635f-0aac-4b69-8b62-c5dd2dc42476
md"""
At cell division we need to partition the protein counts based on the inherited cell size. In particular, we draw a sample p from a Beta distribution corresponding to the inherited size proportion of one of the two daughter cells and sample a value x1 from Binomial(x, β) where x correspond to the protein counts of the mother cell. The difference x2 = x - x1 is then going to be allocated to the second daughter cell. We return a vector of values giving the inherited size proportions of the two daugthers and the sampled protein counts.
"""

# ╔═╡ e67a6092-e8d3-4993-b1f2-8a524cc6434c
begin
	Distributions.Binomial(p::Float64, β::Real) = Distributions.Binomial(round(Int, p), β)

	@register_symbolic Distributions.Binomial(p::Float64, β)
end;

# ╔═╡ 451f3cf4-ede7-411e-9341-9c4d4b18115b
md"""
Using the defined functions the intereaction channel definition becomes.
"""

# ╔═╡ a5a789ea-2760-4815-bb71-9390f0db125c
begin
	@abm_variables C(t)
	@parameters Cs CΔ Cp β B L
	
	division = @interaction begin
	    @channel γdivision($Cs, $CΔ, $α), C --> 2*C
	    @sampler ExtrandeMethod($L; boundtype=:increasing)
		@connections (($CΔ => $Δ, $Cs => $s, $Cp => $p),)		
		@variable $β = rand($Beta(100))
		@variable $B = rand($Binomial($Cp, $β))
	    @transition (
	        ($Δ => 0.0, $s => $β*$Cs, $p => $B),
	        ($Δ => 0.0, $s => (1-$β)*$Cs, $p => $Cp - $B))

	    @saveinstate ($C, $s) ($C, $p)
	    @saveoutstate ($C, $s) ($C, $p)
		@name "division"
	end
end;

# ╔═╡ 1c331c39-4734-44df-bacc-85e21745d420
md"""
We know that the division rate is monotonically increasing. The sampler ExtrandeMethod is passed with the option ```boundtype=:increasing``` to make use of this 
"""

# ╔═╡ 08559fb8-3342-4df5-80d8-4ad280eb472e
md"""
# Step 3
"""

# ╔═╡ b7a54dcb-ecda-4497-b171-fb426b57e90d
md"""
Combining the interaction with the models of internal dynamics.
"""

# ╔═╡ f4281e59-5197-4d6c-90a9-5b298cd1dbec
model = AgentsModel([division, ], Dict(C => Cell,));

# ╔═╡ 36e52938-da75-480d-b7f9-ded770d6ee0d
md"""
Give an initial population of single cell with protein count 0, age 0, size 0.4.
"""

# ╔═╡ fac12c07-df5b-481f-b5da-d79c9db46a56
initial_population = [C => (p => 0.0, s => 0.40, Δ => 0.0), ];

# ╔═╡ eb2d10e3-4872-4f7f-8f8d-2bc5903abea1
md"""
Define some simulation parameters. 
"""

# ╔═╡ ae0cdda0-fdcf-4eae-b342-d2bf4dd5689d
begin
	# Simulation timespan.
	timespan = (0.0, 20.0)
	# Simulator timestep.
	Δt = 1.0
	
	params = SimulationParameters(
	    [α => 1.0, kprod => 10.0, b => 6.0, L => 0.01], timespan, Δt; maxpop=1000)
end;


# ╔═╡ ca19d5ee-e320-4859-91a9-84b85c880317
md"""
Run the simulation. 
+ ```trace_agents```: default ```false``` results in only the final population of agents and their internal simulations being kept in the results structure.
"""

# ╔═╡ e54d3fa5-098b-4010-95fc-78548c4ce9c8
# ╠═╡ show_logs = false
res = simulate(model, initial_population, params);

# ╔═╡ b489fcfc-3759-4f64-aaec-ecae94ac1c3f
md"""
The resulting ```SimulationResults``` structure contains. 

+ ```instates``` --- records the snapshot states of substrate agents at the time of interactions. In this case the protein counts and sizes of the mother cell at division.
+ ```outstates``` --- records the snapshot states of product agents at the time of interactions. In this case the protein counts and sizes with which the daughter cells are intialised at division.
+ ```snapshot``` --- records the defined population snapshots at every simulator timestep.
+ ```agents``` --- when the flag ```trace_agents=true``` is enabled records all agents that have at some point been part of the population along with the their internal simulations. Used for history tracing.
+ ```final_pop``` --- records the agent population present at the final timepoint.
+ ```interactions``` --- when the flag ```save_interactions=true``` is enable records all performed interactions in a tupel consisting of reaction time, performed interaction and the population state after the interaction.
+ ```tend``` --- final time of the simulation.
"""

# ╔═╡ f777fe28-227a-48a6-9b78-16c66a36ffed
md"""
# Plotting
"""

# ╔═╡ 723489bb-7e1a-4146-8aab-efdbddd54eef
md"""
Let's set up the figure for plotting using ```CairoMakie.jl```.
"""

# ╔═╡ df9500b6-14b4-41f9-a1fb-fd25bf88ee6e
begin
	CairoMakie.activate!()
	colors = ColorSchemes.Hiroshige.colors
	transp = 0.8
	htransp = 0.05
	stairstransp = 0.9
	barstransp = 0.3

	pt_cm = 2.83465
	fig = Figure(size=(300, 70) .* pt_cm; fontsize=8, pt_per_unit=1)
	ax_protein = Axis(fig[1,1]; xlabel="Birth protein distribution", ylabel="Probability density")
	ax_size = Axis(fig[1,2]; xlabel="Birth size distribution", ylabel="Probability density")
	hidedecorations!.(fig.content, ticklabels = false, ticks = false, label = false)
	hidespines!.(fig.content, :t, :r) 
	xlims!(ax_protein, (0, 150))

	function plot_hist!(ax, values; bins, color, label="", weight = 1.0, plotbars = true)
		# Function for plotting the histograms with stair outline.
    	hist_ = normalize(fit(Histogram, values, bins); mode=:pdf)
    	stairs!(ax, collect(midpoints(hist_.edges[1])), hist_.weights .* weight;color=(color, stairstransp), step=:center, label=label)
    	plotbars && barplot!(ax, collect(midpoints(hist_.edges[1])), hist_.weights .* weight; 
        color=(color, barstransp), strokecolor=(color, transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)
	end
end;

# ╔═╡ 35268085-c7d1-4683-8b31-2124c825c0e5
md"""
Plot the size and protein distributions at birth.
"""

# ╔═╡ cd707a2c-795a-441d-8b51-019af0f7cfbf

begin
	fig
	plot_hist!(ax_size, reduce(vcat, res.outstates[:Cs_division].u); 
		bins = 0:0.01:1.0, color=colors[1], label = "Agent-based simulation")
	#plot_hist!(ax_protein, 
	#	[Dict(a.init_trait)[p] for a in values(res.final_pop[C])]; 
	#	bins=0:1:120, color=colors[1], label = "Agent-based simulation")
	plot_hist!(ax_protein, 
		reduce(vcat, res.outstates[:Cp_division].u); 
		bins=0:1:120, color=colors[1], label = "Agent-based simulation")
end;

# ╔═╡ 05d96141-f7d0-48e9-9d03-1cc91163fd8b
md"""
Let's compare with analytical computations. These follow [(Thomas and Shahrezaei, 2021)](https://royalsocietypublishing.org/doi/epdf/10.1098/rsif.2021.0274). These computations are not part of the package but can be replicated with the following script (hidden by default).

#Thomas P, Shahrezaei V. 2021 Coordination of gene expression noise
#with cell size: analytical results for agent-based models of growing cell populations. J. R. Soc. Interface 18: 20210274. [10.1098/rsif.2021.0274](https://doi.org/10.1098/rsif.2021.0274)
"""

# ╔═╡ b248142d-c524-4e5a-bc25-fbd61aba240b
begin
	#Parametrisation as in the simulation model.
	α_ = 1.0
	kprod_ = 10.0 
	b_ = 6.0
	
	span_ = (1e-4, 0.8)
	
	intparams = ( 
	    abstol = 1e-4,
	    reltol = 1e-4)
	
	γ(s, s0) = (1 / (α_ * s)) * γdivision(s, s - s0, α_)
	γint(s, s0) = IntegralProblem((u,p) -> γ(u, s0), (s0, s))
	phi(s, s0) = γ(s, s0) * exp(-solve(γint(s, s0), QuadGKJL(); intparams...).u)
	
	ddist = Beta(100)
	
	kernelf(s, s0) = IntegralProblem((u,p) -> 2*pdf(ddist, u) * phi(s/u, s0), span_)
	ker(s, s0) = solve(kernelf(s, s0), QuadGKJL(); intparams...).u
	
	function trapz(fx, xstep)
	    out = 0.0
	    fxx = zip(fx, fx[2:end])
	    for f_ in fxx
	        out += middle(f_...) * xstep 
	    end
	    return out
	end
	
	function volterra(ker, span, n)
	    a, b = span
	    h = (b-a)/n
	    x = range(a, stop=b, length=n)
	    
	    Xi = Float64[]
	    Ai = zeros(length(x), length(x))
	    
	    for i in 1:n
	        for j in range(1, n, step=1)
	            Ai[i,j] = h*ker(x[i], x[j])
	        end
	        Ai[1,i] = h*ker(x[1],x[i])
	        Ai[i,i] = h*ker(x[i],x[i])
	    end
	    return Ai
	end
	
	N = 50
	sstep_ = (span_[2]-span_[1])/N
	srange_ = collect(range(span_[1], stop=span_[2], length=N))
	A = volterra(ker, span_, N)
	sn = trapz(Float64.(eigvecs(A)[:,end]), sstep_)
	
	psiA_tree = Float64.(eigvecs(A)[:,end]) ./ srange_
	sn_tree = trapz(psiA_tree, sstep_)
	
	psi = linear_interpolation(
	    range(span_[1], stop=span_[2], length=N), Float64.(eigvecs(A)[:,end]) .* 1/sn, extrapolation_bc=Line())
	psi_tree_ = linear_interpolation(
	    range(span_[1], stop=span_[2], length=N), psiA_tree .* 1/sn_tree, extrapolation_bc=Line())
	
	# Compute protein distribution for different birth sizes s0.
	function rho(s_, s0_, s0)
	    (1/psi(s0))*(s0/s_)*pdf(ddist, s0/s_)*phi(s_, s0_)*psi(s0_)
	end

	sintspan_ = [span_[1], 2.0]
	# Birth protein counts. Use concentration homeostasis. 
	dist(kprod, b, α, s) = NegativeBinomial(kprod/(α), 1/(1+b*s)) 
	Pi(x, s) = pdf(dist(kprod_, b_, α_, s), x)
	Bpdf(x, x_, θ) = pdf(Binomial(x_, θ), x)
	
	# Integrate
	rhoint(x, x_, s_, s0) = IntegralProblem((u, p) ->  rho(s_, u, s0), sintspan_)
	Pi0int(x, x_, s0) = IntegralProblem(
	    (u,p) -> Bpdf(x, x_, s0/u)*solve(rhoint(x, x_, u, s0), QuadGKJL(); intparams...).u *Pi(x_, u), (s0, sintspan_[end]))
	Pi0(x, x_, s0)  = solve(Pi0int(x, x_, s0), QuadGKJL(); intparams...).u
	
	xs_ = collect(range(0, stop=100, step=5))
	xstep_ = xs_[2] - xs_[1]
	array_dists(x, s0) = map(x_ -> Pi0(x, x_, s0), xs_)
	ss_ = range(sintspan_[1], stop=sintspan_[2], length=20)
	
	mat = [array_dists(x, s) for x in xs_, s in ss_]
	mat_ = hcat([row .* psi_tree_.(ss_) for row in eachrow(mat)]...)'
	mat_ = trapz.(mat_, Ref(xstep_))
	mat_s = trapz.(eachrow(mat_), Ref(sstep_))
	mat_sx = trapz(mat_s, xstep_)
end;

# ╔═╡ 4266ab25-5e95-4241-a257-eb93b6778af1
begin
	# Plot the analytical solutions.
	# Birth size distribution.
	lines!(ax_size, range(span_[1], stop=span_[2], length=N), psi_tree_.(range(span_[1], stop=span_[2], length=N)); color=colors[10], label="Analytical solution")
	# Birth protein distribution.
	lines!(ax_protein, xs_, mat_s ./ mat_sx; color=colors[10], label="Analytical solution")
	#Legend(fig[2,:], ax_protein; orientation=:horizontal)
	xlims!(ax_protein, (0,  100))
	xlims!(ax_size, (0.2,  0.8))
end;

# ╔═╡ 07d7b398-34b8-4198-9f92-eb58763d04bf
fig

# ╔═╡ 6e6e14a3-1199-431c-9716-4e518b251377
md"""
Plot some sample lineages from the simulation. 

The agents and final_pop fields keep track of individual agent in the form 

```julia
Dict(
	symbol1 => Dict(
		uid1 => AgentState,
		uid2 => AgentState
	),
	...
)
```
where agents is a collection of all agents that were part of the population during the simulation and final_pop consists of agents only present at the final timepoint of the simulation.
"""

# ╔═╡ 129a22f2-e3d8-41a4-be59-fe761256372e
md"""
The AgentState structure gives access to the following fields

```julia
mutable struct AgentState{tType, biType, sType, pType, inType, cType}
    sym::sType # Symbol of representing the agent type.
    btime::tType # Time agent was created (birth time).
    dtime::tType # Time agent was destroyed.
    binteraction::biType # Interaction name resulted in the creaction of the agent (if specified in the interactions).
    parents::pType # Parent of the agent. Each parent given as tuple (sym, uid).
    uid::UInt # Unique identifier.
    init_trait::inType # Initial values of the traits.
    consts::cType # Values of constants. 
    simulation::Union{Nothing, ODESolution, RODESolution} # State simulation results.
end
```
"""

# ╔═╡ e8bd592d-079d-46f6-a746-507044f46f90

md"""
The parent identifiers can then be used to trace the lineage history of the agents. We have implemented a helper function
```julia
AgentBasedModeling.lineage(res, agent)
```
taking as aguments the results structure and an AgentState and returing a list of agents in the lineage history of the provided agent. This helper assumes each agent has a single parent.   
"""

# ╔═╡ 14bf24fc-5339-4531-8388-fc3d1e640c9d
begin
	fig_traj = Figure(size=(300, 80) .* pt_cm; fontsize=8, pt_per_unit=1)
	ax_protein_traj = Axis(fig_traj[1,1]; xlabel="Protein lineage trajectories", ylabel="Probability density")
	ax_size_traj = Axis(fig_traj[1,2]; xlabel="Size lineage trajectories", ylabel="Time")
	hidedecorations!(ax_protein_traj, ticklabels = false, ticks = false, label = false)
	hidespines!(ax_protein_traj, :t, :r) 

	hidedecorations!(ax_size_traj, ticklabels = false, ticks = false, label = false)
	hidespines!(ax_size_traj, :t, :r) 
end;

# ╔═╡ e1c44637-92e8-4d9c-b10b-0e91cbc39e69
md"""
AgentBasedModeling.jl provides a function ```lineage``` that extract the history of an agent from the population. 
"""

# ╔═╡ eb66a606-b38f-4044-b772-436755d430d5
begin
	function plot_lineage(axsize, axprotein, agent; kwargs...)
		# Get the lineage history of a cell.
	    lin = AgentBasedModeling.lineage(res, agent)
	    for cell in lin
			# Plot trajectories of each cell.
	        sim = cell.simulation
	        if !isnothing(cell.dtime) & !isinf(cell.dtime)
				# Cells that divided.
				plottspan = cell.btime:0.1:cell.dtime
				# Plot size s.
	            lines!(axsize, sim(plottspan).t, sim(plottspan; idxs=s).u; kwargs...)
				# Plot protein counts p.
	            lines!(axprotein, sim(plottspan).t, sim(plottspan; idxs=p).u; kwargs...)
	        else
				# Cells still alive at the end of the simulation.
				plottspan = cell.btime:0.1:sim.t[end]
	            lines!(axsize, sim(plottspan).t, sim(plottspan; idxs=s).u; kwargs...)
	            lines!(axprotein, sim(plottspan).t, sim(plottspan; idxs=p).u; kwargs...)
	        end
	    end
	end

	i = 1
	# Lineages of random 50 cells from the final population as shaded lines.
	for agent in rand(collect(values(res.final_pop[C])), 50)
	    plot_lineage(ax_size_traj, ax_protein_traj, agent; color = (colors[i], htransp), linewidth=0.8)
	    global i = mod(i + 1, length(colors)) + 1
	end

	# Random lineage as solid lines.
	agent = rand(collect(values(res.final_pop[C])))
	plot_lineage(ax_size_traj, ax_protein_traj, agent; color = colors[1], linewidth=1.2)
	fig_traj
end

# ╔═╡ c4549ac3-c826-4caa-864d-835e43a459fd
md"""
Finally, lets produce a visualisation of the interaction graph where nodes represent interactions and edges agents.
"""

# ╔═╡ 9feec140-a725-4253-ab69-c13b8f2454f8
md"""
Helper functions for extracting the protein counts and constructing the x, y coordinates of vertices. 
"""

# ╔═╡ ec3e1785-7e29-4d27-9572-c9397fa630ff
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
end;

# ╔═╡ 1736b534-6f00-43bb-bcfc-897a132c9b52
md"""
Using the ```construct_interaction_graph``` from ```AgentBasedModeling.jl``` to construct the graph.
"""

# ╔═╡ dee1cc50-90f5-48fd-96bc-fb0c611e7c3f
itx_graph = construct_interaction_graph(res);

# ╔═╡ 31e98d17-23a5-4fc8-b109-281c8545abde
md"""
Plotting.
"""

# ╔═╡ 2fad00bd-69d5-4420-897b-e29b809f563d
begin	
	fig_tree = Figure(size=(300, 80) .* pt_cm; fontsize=8, pt_per_unit=1)
	ax = Axis(fig_tree[1,2], xlabel="Time")
	hidedecorations!.(fig_tree.content, ticks = false, label = false, ticklabels=false)
	hideydecorations!.(fig_tree.content)
	hidespines!.(fig_tree.content, :t, :r, :l) 
	
	layout = make_layout(itx_graph)
	
	for edge in edges(itx_graph)
		data_ = edge_data(itx_graph, edge)
		src_ = edge.src 
    	dst_ = edge.dst 
	
	    lines!(ax, [layout[src_], layout[dst_]]; color=data_, colorrange=(10.0, 60.0), colormap=range(colors[7], stop=colors[1], length=10), linewidth=2)
	end
	scatter!(ax, layout; color=:black, markersize=5)
	xlims!(ax, high=3.7)
	
	Colorbar(fig_tree[1, 1], limits = (10, 60), colormap=range(colors[7], stop=colors[1], length=10), flipaxis = false, label="Protein counts")
	resize_to_layout!(fig)
	fig_tree;
end

# ╔═╡ Cell order:
# ╟─4e3e20d3-d152-4c26-a1f8-a0f4d11cf025
# ╟─696a96c3-3499-41a7-a045-ec5d55806319
# ╠═355d1e6c-91f6-11ef-2f45-19eb05b5141f
# ╟─93633661-a134-4f9d-89f3-497c38283181
# ╟─019b50c4-dc51-45eb-bc28-770b8d5c2696
# ╠═35c0a480-6582-4531-ae36-1135665ee564
# ╟─d6827f12-e43b-4f19-bfa0-c8cc5d040fa4
# ╟─1759ae4c-60ec-493d-a2d4-06b7cc83a8f2
# ╠═fee337cd-3f0c-4ccd-a023-7aa4e717c678
# ╟─7fcf8886-a1f9-419f-aef7-18ab5a39ce52
# ╟─6f00bd54-ffd2-4f41-a533-a35727ba590c
# ╠═a5ff7927-087b-43bb-80a9-d9d13c3fe2c5
# ╟─7883635f-0aac-4b69-8b62-c5dd2dc42476
# ╠═e67a6092-e8d3-4993-b1f2-8a524cc6434c
# ╟─451f3cf4-ede7-411e-9341-9c4d4b18115b
# ╠═a5a789ea-2760-4815-bb71-9390f0db125c
# ╟─1c331c39-4734-44df-bacc-85e21745d420
# ╟─08559fb8-3342-4df5-80d8-4ad280eb472e
# ╟─b7a54dcb-ecda-4497-b171-fb426b57e90d
# ╠═f4281e59-5197-4d6c-90a9-5b298cd1dbec
# ╠═36e52938-da75-480d-b7f9-ded770d6ee0d
# ╠═fac12c07-df5b-481f-b5da-d79c9db46a56
# ╟─eb2d10e3-4872-4f7f-8f8d-2bc5903abea1
# ╠═ae0cdda0-fdcf-4eae-b342-d2bf4dd5689d
# ╟─ca19d5ee-e320-4859-91a9-84b85c880317
# ╠═e54d3fa5-098b-4010-95fc-78548c4ce9c8
# ╟─b489fcfc-3759-4f64-aaec-ecae94ac1c3f
# ╟─f777fe28-227a-48a6-9b78-16c66a36ffed
# ╟─723489bb-7e1a-4146-8aab-efdbddd54eef
# ╠═df9500b6-14b4-41f9-a1fb-fd25bf88ee6e
# ╟─35268085-c7d1-4683-8b31-2124c825c0e5
# ╠═cd707a2c-795a-441d-8b51-019af0f7cfbf
# ╠═05d96141-f7d0-48e9-9d03-1cc91163fd8b
# ╟─b248142d-c524-4e5a-bc25-fbd61aba240b
# ╠═4266ab25-5e95-4241-a257-eb93b6778af1
# ╠═07d7b398-34b8-4198-9f92-eb58763d04bf
# ╠═6e6e14a3-1199-431c-9716-4e518b251377
# ╟─129a22f2-e3d8-41a4-be59-fe761256372e
# ╟─e8bd592d-079d-46f6-a746-507044f46f90
# ╠═14bf24fc-5339-4531-8388-fc3d1e640c9d
# ╟─e1c44637-92e8-4d9c-b10b-0e91cbc39e69
# ╠═eb66a606-b38f-4044-b772-436755d430d5
# ╟─c4549ac3-c826-4caa-864d-835e43a459fd
# ╟─9feec140-a725-4253-ab69-c13b8f2454f8
# ╠═ec3e1785-7e29-4d27-9572-c9397fa630ff
# ╟─1736b534-6f00-43bb-bcfc-897a132c9b52
# ╠═dee1cc50-90f5-48fd-96bc-fb0c611e7c3f
# ╟─31e98d17-23a5-4fc8-b109-281c8545abde
# ╠═2fad00bd-69d5-4420-897b-e29b809f563d
