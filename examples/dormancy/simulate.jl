using FileIO
using OrdinaryDiffEq
using CSV
using DataStructures
using StatsBase
using Base.Threads
using ProgressMeter

include("model.jl")

tspan = (0.0, 100.0)
maxN = 100
replN = 5000
# Trajectory saving timestep.
Δt = 1.0

mkpath("$(datadir())/dormancy/")

# Initial population state. Starting with a single cell at age 0.
init_pop = repeat([C => (τ => 0.0, )], 1)

α = 0.1

# Dictionary for the parameters.
fixed_params = Dict(
    L => 0.5,
    T1 => 1.0, T2 => 1.0, 
    αdiv => 1.8, βdiv => 1.0, 
    αdeath => 2.0, βdeath => 1e-3, 
    αendorm => α, βendorm => 1.0,
    αexdorm => α, βexdorm => 1.0, 
)

vary_T = exp.(range(-1.0, stop=3.0, length=10)) 

filename = "$(datadir())/dormancy/results_dormancy_muN$(α).csv"
file = open(filename,"w")

for params in vary_T
    p = Progress(replN)
    display("T = $params")
    fixed_params[L] = min(0.99*params, 0.5)
    fixed_params[T1] = params
    fixed_params[T2] = params
    # snapshot part tells the simulator to take a snapshot of the number of
    # agents of type C every Δt. 
    simulation_params = SimulationParameters(collect(fixed_params), tspan, Δt, Tsit5(); 
        maxpop=maxN, 
        snapshot=[PopulationSnapshot(C), PopulationSnapshot(D)])
    ext = Vector{Bool}(undef,replN)
    telaps = @elapsed @threads for i = 1:replN
        res = simulate(population_model, init_pop, simulation_params; showprogress=false)
        ext[i] = res.snapshot[:C][end].values != 0 || res.snapshot[:D][end].values != 0
        next!(p)
    end
    finish!(p)
    cmp = countmap(ext)
    def_cmp = DefaultDict(0, cmp)
    save_row = [params..., def_cmp[0], def_cmp[1]]
    write(file, join(save_row, ","), "\n")
    println("Time taken $telaps.")
end

close(file)
