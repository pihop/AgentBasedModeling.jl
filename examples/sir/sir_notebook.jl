### A Pluto.jl notebook ###
# v0.20.17

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
		Pkg.PackageSpec(name="ProgressMeter"),		Pkg.PackageSpec(name="OrdinaryDiffEq"),
		Pkg.PackageSpec(name="JumpProcesses"),
		Pkg.PackageSpec(url="https://github.com/pihop/AgentBasedModeling.jl")
    ])
	using AgentBasedModeling
	using Catalyst
	using ModelingToolkit
	using Distributions
	using LinearAlgebra
	using Interpolations
	using OrdinaryDiffEq
	using CairoMakie
	using StatsBase
	using ColorSchemes
	using Colors
	using GraphMakie
	using ProgressMeter
	using JumpProcesses
	import Distributions: Categorical
end;

# ╔═╡ ea9c5a6d-7225-43b7-b3f9-562f2185f5b0
md"""
# Susceptible-infected-recovered model 
"""

# ╔═╡ 6b3dab0d-b5c5-4632-a7f2-43fa0a07b01f
md"""
# Individual state dynamics
"""

# ╔═╡ 2cdf4d9f-0374-4bb6-bf7f-f34361d1a6b7
md"""
We define the state dynamics of the Susceptible(S), Infected(I) and Recovered(R) individuals. Infected individuals state is caputered by the time since infection which will later be used to model incubation times of the disease. Each individual will have a constant age group (Age) that is going to model the differing susceptibility of individuals in different age groups.
"""

# ╔═╡ 1edcea5a-cb1e-4f26-b9c7-d8227d17173d
begin
	@independent_variables t
	@species I(t), S(t), R(t), τ(t)
	@parameters Age
	D = Differential(t)

	@named InfectedDyn = ODESystem([D(τ) ~ 1.0, ], t)
	@named SusceptibleDyn = EmptyTraitProblem() 
	@named RecoveredDyn = EmptyTraitProblem()
	
	Infected = AgentDynamics(InfectedDyn, (Age,))
	Susceptible = AgentDynamics(SusceptibleDyn, (Age,))
	Recovered = AgentDynamics(RecoveredDyn, (Age,))
end;

# ╔═╡ a8cff812-98ae-42cc-b1de-0130b5326f12
md"""
# Interactions
"""

# ╔═╡ b549d1a5-4e90-417a-9890-7ad682f7f6af
md"""
## Infection
"""

# ╔═╡ a5e29600-e029-4551-9754-edd0322a09e3
begin
	@parameters Iτ, β, K, n, SAge, IAge, LK
	
	function γ_infect(τ, β, K, n, Age) 
	    β_ = Age == 1 ? β / 2 : β
	    return (β_ / (1 + (K / τ)^n))
	end
	@register_symbolic γ_infect(τ, β, K, n, Age) 	
	
	infection = @interaction begin
	    @channel γ_infect($Iτ, $β, $K, $n, $SAge) / ($S + $I + $R), $I + $S --> 2 * $I
	    @sampler ExtrandeMethod($LK, boundtype=:increasing)
	    @connections (
	        ($Iτ => $τ, $IAge => $Age),
	        ($SAge => $Age, )
	    )
	    @transition (($τ => $Iτ, $Age => $IAge), ($τ => 0.0, $Age => $SAge))
	    @savesubstrates ($I, $τ)
	end
end;

# ╔═╡ 03dba07e-a5b9-45cc-9f6e-1572a3bc5fb0
md"""
Function ```γ_infect``` defines the rate of infection interaction between a susceptible and infected individual and depends on the time since the infected individial got the disease and the age group of the susceptible individual. Age group 2 is more likly to get the disease than age group 1. 
"""

# ╔═╡ c3e142d8-9a28-4fea-84e4-34dccd37f380
md"""
## Recovery
"""

# ╔═╡ f558c21e-81a3-4dea-bca0-c6f25143fb33
begin
	@parameters γ
	γ_recover(τ) = γ
	recovery = @interaction begin
	    @channel γ_recover($Iτ), $I --> $R 
	    @sampler GillespieMethod()
	    @connections (($Iτ => $τ, $IAge => $Age,),)
	    @transition (($Age => $IAge, ), )
	    @savesubstrates ($I, $τ)
	end
end;

# ╔═╡ 8105db17-6d04-416e-b682-ff7984ec32a0
md"""
Recovery happens at a constant rate.
"""

# ╔═╡ 8ffc3539-afcb-46d0-94ee-a0386184774f
md"""
## Immigration
"""

# ╔═╡ 2a023bd9-7718-44d1-b7c8-9254ba6d564d
md"""
New susceptible individuals can enter into the system. Their age groups are initialised by sampling a categorical distribution.
"""

# ╔═╡ d0f41377-1a6c-4644-bb73-2d8857c2693d
begin
	@parameters μ, a_
	# Two age groups 0-20, 20-
	agedist = Categorical([0.24, 0.76])
	function sample_age()
	    rand(agedist)
	end

	immigration = @interaction begin
	    @channel $μ*($S + $I + $R), 0 --> $S
	    @variable $a_ = $sample_age() 
	    @transition (($τ => 0.0, $Age => $a_), )
	    @sampler ExtrandeMethod($μ*($S+$I+$R), Inf; trait_indep=true)
	end
end;

# ╔═╡ c8e5cb80-abce-42a4-a270-f198c2ea9749
md"""
## Emigration
individuals can leave the system.
"""

# ╔═╡ 768f104e-cf18-4a91-b24c-2cbdb7723245
begin
	s_emig = @interaction begin
	    @channel $μ, $S --> 0 
	    @sampler GillespieMethod()
	end
	
	i_emig = @interaction begin
	    @channel $μ, $I --> 0 
	    @sampler GillespieMethod()
	end
	
	r_emig = @interaction begin
	    @channel $μ, $R --> 0 
	    @sampler GillespieMethod()
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

# ╔═╡ 5085eaac-876c-4b37-a88a-7dc0eb05ef9c
begin
	# Put them together.
	behaviours = Dict(I => Infected, S => Susceptible, R => Recovered)
	population_model = AgentsModel(
		[infection, recovery, immigration, s_emig, i_emig, r_emig], behaviours)
end;

# ╔═╡ a24f9ead-8043-4bf9-9874-4b36c4be2c49
md"""
# Simulate
"""

# ╔═╡ 2e5b9514-e9b0-4f6a-8960-92ef6d642ff7
md"""
Helper function that sets the simulation parameters and aggregates the results to an Ensemble
"""

# ╔═╡ b71e807d-8c70-4d45-9067-1d44088b10fc
function simulate_bulk(nsim, Npop, I0, mparams, tspan, Δt; model)
	p = Progress(nsim)
  
	init_traits = (τ => 0.0, Age => sample_age())
	init_pop = (repeat([I,], I0) .=> ((τ => 0.0, Age => sample_age()) for _ in 1:I0),
				repeat([S,], Npop-I0) .=> ((Age => sample_age(), ) for _ in 1:(Npop-I0)))
	init_pop = vcat(init_pop...)

	simulation_params = SimulationParameters(
		mparams, tspan, Δt, Tsit5(); 
		snapshot=[
			PopulationSnapshot(R), 
			PopulationSnapshot(I), 
			PopulationSnapshot(S),
			StateSnapshot(I, τ)])


	println("Simulation started")
	telapse = @elapsed begin
		solns = Vector(undef, nsim)
        # For multithreading
#		Threads.@threads for i in 1:nsim
		for i in 1:nsim
			res = simulate(model, init_pop, simulation_params; showprogress=false)
			solns[i] = AgentBasedModeling.build_snapshot_solution(
				res.snapshot; names=[:I, :S, :R])
			next!(p)
		end
	end
	finish!(p)

	return EnsembleSolution(solns, telapse, true)
end;

# ╔═╡ ec1daa94-8c0d-4a73-a61e-9faae71fa966
begin
	R0_ = 1.5 
	ε_ = 0.04
	μ_ = 0.02
	γ_ = (μ_ / ε_) - μ_

	# Model parameters
	mparams = Dict(
		K => 0.15,
		n => 3.0,
		LK => 1.0,
		γ => γ_,
		β => R0_ / (γ_ + μ_)
	)

	# Simulator parameters.
	Δt = 1.0
	tspan = (0.0, 100.0)
	I0 = 3 

	simres = simulate_bulk(100, 200, 3, mparams, tspan, Δt; model=population_model)
end;

# ╔═╡ da7ef7e1-8dee-4bf3-bef3-e7326f0e03da
md"""
## Standard SIR model

We implement the standard SIR model using Catalyst.jl.
"""

# ╔═╡ 7fd75040-894d-41e9-8f51-0c3b3f9add9d
begin
	N = 200
	standard_sir = @reaction_network begin
	    @parameters μ γ β
	    @species S(t) I(t) R(t)
	    μ*(S+I+R), 0 --> S
	    μ, S --> 0
	    μ, I --> 0
	    μ, R --> 0
	    β/(S+I+R), S + I --> 2 * I
	    γ, I --> R
	end
	p  = (:μ => μ_, :γ => γ_, :β => R0_ / (γ_ + μ_))
	u0 = [:I => I0, :S => N - I0, :R => 0]
	dprob = DiscreteProblem(standard_sir, u0, tspan, p)
	jprob = JumpProblem(standard_sir, dprob, Direct())
	eprob = EnsembleProblem(jprob)
	esol = solve(eprob, SSAStepper(), EnsembleThreads(); trajectories = 10000)
	esum = EnsembleSummary(esol, tspan[1]:Δt:tspan[2]; )
end;

# ╔═╡ 9586b926-0309-41d2-a96e-97ba104104c7
md"""
# Plotting
"""

# ╔═╡ 25ae1ba3-600d-4f12-b6a5-142a54a2c275
md"""
Extrat the extinction times.
"""

# ╔═╡ 4f0b4d84-94ec-4bff-9ae9-58f6ac521a97
begin
	extimes_agentsim = []
	for r in simres.u
	    idx_ = findfirst(x -> x[1] == 0, r.u)
	    isnothing(idx_) && continue
	    push!(extimes_agentsim, r.t[idx_]) 
	end

	times_cmesim = [] 
	for r in esol
	    idx_ = findfirst(x -> x[2] == 0, r.u)
	    isnothing(idx_) && continue
	    push!(times_cmesim, r.t[idx_]) 
	end
end

# ╔═╡ e9ece2ae-5994-4d3e-b521-be87b8198e14
md"""
Plot the extinction times.
"""

# ╔═╡ f731b8c4-1e58-42d1-b7a9-beff535893e1
begin
	colors = ColorSchemes.Hiroshige.colors
	transp = 0.2
	stairstransp = 0.9

	pt_cm = 2.83465
	# 1 cm = 28,3465 pt
	# Total size 170x225mm
	fig = Figure(size=(170, 50) .* pt_cm; fontsize=8, pt_per_unit=1, figure_padding = 0.1)
	
	axt = Axis(fig[1,2]; xlabel="Time to extinction", ylabel="Proabability density", title="Burnout time")
	axtraj = Axis(fig[1,1]; 
	    xlabel = "Time", 
	    ylabel = "Infected population size"
	    )
	
	ext_cme = normalize(fit(Histogram, times_cmesim, 1:2:100); mode=:pdf)
	ext_agent = normalize(fit(Histogram, extimes_agentsim, 1:2:100); mode=:pdf)
	
	stairs!(axt, collect(midpoints(ext_cme.edges[1])), ext_cme.weights; color=(colors[1], stairstransp), step=:center, label="Gillespie")
	barplot!(axt, collect(midpoints(ext_cme.edges[1])), ext_cme.weights; 
	    color=(colors[1], transp), strokecolor=(colors[1], transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)
	
	stairs!(axt, collect(midpoints(ext_agent.edges[1])), ext_agent.weights; color=(colors[10], stairstransp), step=:center, label="Agent-based model")
	barplot!(axt, collect(midpoints(ext_agent.edges[1])), ext_agent.weights; 
	    color=(colors[10], transp), strokecolor=(colors[10], transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)
	
	hidedecorations!(axt, ticks=false, label=false, ticklabels=false)
	hidespines!(axt, :r, :t)
	axislegend(axt; orientation=:vertical, framevisible=false, tellwidth=false, tellheight=false, rowgap = -10, labelsize=6)
	
	xlims!(axt, (1, 100))
	ylims!(axt, (0, 0.1))

	for (i,idx) in zip([5,1], [5,100])
		traj = simres.u[idx]
	    lines!(axtraj, traj.t, getindex.(traj.u, 1); color=colors[i], linewidth=2.0)
	    idx_ = findfirst(x -> x == 0, getindex.(traj.u, 1))
	    scatter!(axtraj, traj.t[idx_], 0; color=colors[i], markersize=10, strokecolor=colors[i], strokewidth=0.5)
	end
	
	hidedecorations!(axtraj, ticks=false, label=false, ticklabels=false)
	hidespines!(axtraj, :r, :t)

	fig
end

# ╔═╡ Cell order:
# ╟─ea9c5a6d-7225-43b7-b3f9-562f2185f5b0
# ╠═fd7efc3c-ff22-11ef-34c6-dfce6f883a42
# ╟─6b3dab0d-b5c5-4632-a7f2-43fa0a07b01f
# ╟─2cdf4d9f-0374-4bb6-bf7f-f34361d1a6b7
# ╠═1edcea5a-cb1e-4f26-b9c7-d8227d17173d
# ╟─a8cff812-98ae-42cc-b1de-0130b5326f12
# ╟─b549d1a5-4e90-417a-9890-7ad682f7f6af
# ╠═a5e29600-e029-4551-9754-edd0322a09e3
# ╟─03dba07e-a5b9-45cc-9f6e-1572a3bc5fb0
# ╟─c3e142d8-9a28-4fea-84e4-34dccd37f380
# ╠═f558c21e-81a3-4dea-bca0-c6f25143fb33
# ╟─8105db17-6d04-416e-b682-ff7984ec32a0
# ╟─8ffc3539-afcb-46d0-94ee-a0386184774f
# ╟─2a023bd9-7718-44d1-b7c8-9254ba6d564d
# ╠═d0f41377-1a6c-4644-bb73-2d8857c2693d
# ╟─c8e5cb80-abce-42a4-a270-f198c2ea9749
# ╠═768f104e-cf18-4a91-b24c-2cbdb7723245
# ╟─0b6e84ce-c153-45c8-b382-da4b80be4d17
# ╟─8cd41d0c-aa0a-422c-85b6-8642cd492652
# ╠═5085eaac-876c-4b37-a88a-7dc0eb05ef9c
# ╟─a24f9ead-8043-4bf9-9874-4b36c4be2c49
# ╟─2e5b9514-e9b0-4f6a-8960-92ef6d642ff7
# ╠═b71e807d-8c70-4d45-9067-1d44088b10fc
# ╠═ec1daa94-8c0d-4a73-a61e-9faae71fa966
# ╟─da7ef7e1-8dee-4bf3-bef3-e7326f0e03da
# ╠═7fd75040-894d-41e9-8f51-0c3b3f9add9d
# ╟─9586b926-0309-41d2-a96e-97ba104104c7
# ╟─25ae1ba3-600d-4f12-b6a5-142a54a2c275
# ╠═4f0b4d84-94ec-4bff-9ae9-58f6ac521a97
# ╟─e9ece2ae-5994-4d3e-b521-be87b8198e14
# ╠═f731b8c4-1e58-42d1-b7a9-beff535893e1
