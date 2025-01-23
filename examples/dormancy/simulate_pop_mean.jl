using FileIO
using OrdinaryDiffEq
using CSV
using DataStructures
using StatsBase
using Base.Threads
using ProgressMeter
using CairoMakie

include("model.jl")

tspan = (0.0, 1000.0)
maxN = 300
replN = 5000
# Trajectory saving timestep.
Δt = 1.0

mkpath("$(datadir())/dormancy/")

# Initial population state. Starting with a single cell at age 0.
init_pop = repeat([C => (τ => 0.0, )], 1)

# Dictionary for the parameters.
mus = [0.1, 0.5, 1.0, 10.]

fixed_params = Dict()
for mu in mus
    fixed_params[mu] = Dict(
        L => 0.5,
        T1 => 0.3, T2 => 0.3, 
        αdiv => 4.0, βdiv => 1.0, 
        αdeath => 2.0, βdeath => 1e-3, 
        αendorm => mu, βendorm => 1.0,
        αexdorm => mu, βexdorm => 1.0)
end
#mug = 4, CV_g^2 = 1
#mud = 2, CV_d^2 = 10^-3
#mu_off = mu_on = 2
#CV_off^2 = CV_on^2 = 1
#T_on = T_off = 0.3

function simulate_bulk(ntraj, params; model)
    params_ = params
    p = Progress(ntraj)
    params_[L] = min(0.99*min(params_[T1], params_[T2]), 0.5)

    simulation_params = SimulationParameters(collect(params_), tspan, Δt, Tsit5(); 
        maxpop=maxN, 
        snapshot=[PopulationSnapshot(C), PopulationSnapshot(D)])

    println("Simulation started")
    telapse = @elapsed begin
        solns = Vector(undef, ntraj)
        Threads.@threads for i in 1:ntraj
            res = simulate(model, init_pop, simulation_params; showprogress=false)
            solns[i] = AgentBasedModeling.build_snapshot_solution(res.snapshot; names=[:C, :D])
            next!(p)
        end
    end
    finish!(p)

    sol = EnsembleSolution(solns, telapse, true)
    sum = EnsembleSummary(sol, 0:1.0:1000)

    mu = params_[αendorm]
    CSV.write("trajectory_$(mu)_$(maxN).csv", DataFrame(t=esumdict[mu].u.t, u1=getindex.(esumdict[mu].u.u, 1), u2=getindex.(esumdict[mu].u.u, 2)))
end

for mu in mus 
    simulate_bulk(5000, fixed_params[mu]; model=population_model)   
end

resdict = Dict(mu => simulate_bulk(5000, fixed_params[mu]; model=population_model) for mu in mus)
esumdict = Dict(mu => EnsembleSummary(resdict[mu], 0:1.0:1000) for mu in mus)

