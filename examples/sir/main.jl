using OrdinaryDiffEq
using CairoMakie
using ProgressMeter
using JumpProcesses

using ColorSchemes
colors = ColorSchemes.Hiroshige.colors
transp = 0.2

include("model.jl")

# Model parameters
R0_ = 1.5
ε_ = 0.04
μ_ = 0.02
γ_ = μ_/ε_ - μ_
β_ = R0_ / (γ_ + μ_)

# Initial Population 
K_ = 0.15
n_ = 3.0

# Simulator parameters.
Δt = 1.0
tspan = (0.0, 20.0)
max_rxn = 1e12
I0 = 3 

# Initial population state. Starting with a single cell at age 0.

function simulate_bulk(n, N, lookahead; model)
    p = Progress(n)
  
    init_traits = (τ => 0.0, Age => sample_age())
    init_pop = (repeat([I,], I0) .=> ((τ => 0.0, Age => sample_age()) for _ in 1:I0),
                repeat([S,], N-I0) .=> ((Age => sample_age(), ) for _ in 1:(N-I0)))
    init_pop = vcat(init_pop...)

    simulation_params = SimulationParameters(
        (K => K_, n => n_, μ => μ_, γ => γ_, β => β_, LK => lookahead), tspan, Δt, Tsit5(); 
        snapshot=[
            PopulationSnapshot(R), 
            PopulationSnapshot(I), 
            PopulationSnapshot(S),
            TraitSnapshot(I, τ)])


    println("Simulation started")
    telapse = @elapsed begin
        solns = Vector(undef, n)
        Threads.@threads for i in 1:n
#        for i in 1:n
            res = simulate(model, init_pop, simulation_params; showprogress=false)
            solns[i] = AgentBasedModeling.build_snapshot_solution(res.snapshot; names=[:I, :S, :R])
            next!(p)
        end
    end
    finish!(p)

    return EnsembleSolution(solns, telapse, true)
end

function simulate_traj(n, N, lookahead; model)
    p = Progress(n)
  
    I0 = 3 
    init_traits = (τ => 0.0, Age => sample_age())
    init_pop = (repeat([I,], I0) .=> ((τ => 0.0, Age => sample_age()) for _ in 1:I0),
                repeat([S,], N-I0) .=> ((Age => sample_age(), ) for _ in 1:(N-I0)))
    init_pop = vcat(init_pop...)

    simulation_params = SimulationParameters(
        (K => K_, n => n_, μ => μ_, γ => γ_, β => β_, LK => lookahead), tspan, Δt, Tsit5(); 
        snapshot=[
            PopulationSnapshot(R), 
            PopulationSnapshot(I), 
            PopulationSnapshot(S),
            TraitSnapshot(I, τ)])

    println("Simulation started")
    telapse = @elapsed begin
        solns = Vector(undef, n)
        Threads.@threads for i in 1:n
            res = simulate(model, init_pop, simulation_params; showprogress=n==1)
            solns[i] = res
            next!(p)
        end
    end
    finish!(p)

    return solns
end

N = 1500
lookahead = 1.0
res = simulate_bulk(10000, N, lookahead; model=population_model)
save("$(datadir())/sir/sir_results.jld2", "ares", res)

res_traj = simulate_traj(10, 1000, lookahead; model=population_model_L)
save("$(datadir())/sir/trajectories.jld2", "trajectories", res_traj)
