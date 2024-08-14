using AgentBasedModeling
using OrdinaryDiffEq
using ModelingToolkit
using Catalyst
using Distributions
using StatsBase
using FileIO
using CSV
using DataFrames
using DataStructures
#using SpecialFunctions
#using CairoMakie
using Random 
using Base.Threads
using ProgressMeter

# Defines the variables for population counts C and ages τ.
@variables t 
@species C(t) τ(t)
# Defines the parameters that are going to be used. Cτ is the only funny one --
# we're going to use it as link between the trait and division dynamics layers.
@parameters Cτ, Ton, Toff, αdiv, βdiv, αdeath, βdeath, L
D = Differential(t)

# Define the gamma hazard.
function gammahaz(α,β,x)
    try 
        return exp(logpdf(Gamma(1/β, α*β), x) - logccdf(Gamma(1/β, α*β), x))
    catch
#        @warn "Numerical limits of Gamma hazard" 
        return 1/(α*β)
    end    
end

function death_rate(T1, T2, α, β, age, time) 
    T = T1 + T2 
    if mod(time, T) <= T1
        return gammahaz(α, β, age)
    else
        return 0.0
    end
end

function death_bound(T1, T2, α, β, age, time, L) 
    T = T1 + T2 
    if mod(time, T) ≤ T1 && mod(time + L, T) > T1
        diff = T1 - mod(time, T) 
        return gammahaz(α, β, age + diff) 
    else 
        return death_rate(T1, T2, α, β, age + L, time + L)
    end
end

function birth_bound(α, β, age, L) 
    aB = gammahaz(α, β, age)
    return aB + 1e-8
end

@register_symbolic gammahaz(α,β,x) 
@register_symbolic death_rate(Ton, Toff, α, β, age, time) 
@register_symbolic death_bound(Ton, Toff, α, β, age, time, L) 
@register_symbolic birth_bound(α, β, age, L) 

# Age increases linearly in time t.  
@named CellAge = ODESystem([D(τ) ~ 1.0, ], t)
Cell = AgentDynamics([CellAge,], ())

# Interactions composed of:
# population reaction definition.
# parameter connections. In this case Cτ of the population interaction
# definitions is a variable τ in the agents of type C. 
# how traits are initialised after population interactions.
# sampler, a bound function for the hazard is given symbolically.

cell_division = @population_itx begin
    @channel gammahaz($αdiv, $βdiv, $Cτ), $C --> 2*$C
    @sampler FirstReactionMethod($birth_bound($αdiv, $βdiv, $Cτ, $L), $L)
    @connections ($Cτ, $C, $τ)
    @transition ($τ => 0.0, ), ($τ => 0.0, )
    @savesubstrates ($C, $τ)
end

cell_death = @population_itx begin
    @channel death_rate($Ton, $Toff, $αdeath, $βdeath, $Cτ, $t), $C --> 0
    @sampler FirstReactionMethod($death_bound($Ton, $Toff, $αdeath, $βdeath, $Cτ, $t, $L), $L)
    @connections ($Cτ, $C, $τ)
    @transitions ()
    @savesubstrates ($C, $τ)
end


# Put them together.
rxs = [cell_death, cell_division]
population_model = PopulationModelDef(rxs, Dict(C => Cell))

# Initial population state. Starting with a single cell at age 0.
init_pop = repeat([C => (τ => 0.0, )], 1)

# Dictionary for the parameters.
fixed_params = Dict(
    L => 0.1,
    Ton => 0.1, 
    Toff => 0.1, 
    αdiv => 1.8, βdiv => 1.0, 
    αdeath => 2.0, βdeath => 1e-3)

vary_T_1 = exp.(range(-2.0, stop=-0.8, length=10))
vary_T_2 = exp.(range(-0.85, stop=1.7, length=10)) 
vary_T = vcat(vary_T_1, vary_T_2)

tspan = (0.0, 100.0)
maxN = 101
replN = 20000
# Trajectory saving timestep.
Δt = 1.0

mkpath("$(datadir())/two_env/")
filename = "$(datadir())/two_env/results.csv"
file = open(filename,"w")

#For each parametrisation simulate trajectories and check whether or not the population has survived by the end of the
#simulation.
for params in vary_T[2:4]
    p = Progress(replN)
    display("T = $params")
    fixed_params[Ton] = 0.5*params
    fixed_params[Toff] = params
    simulation_params = SimulationParameters(collect(fixed_params), tspan, Δt, Tsit5(); 
        maxpop=maxN, 
        snapshot=[PopulationSnapshot(C),])
    ext = Vector{Bool}(undef,replN)
    telaps = @elapsed for i = 1:replN
        res = simulate(population_model, init_pop, simulation_params; showprogress=false)
        ext[i] = res.snapshot[:C][end].values != 0   
        next!(p)
    end
    finish!(p)
    cmp = countmap(ext)
    def_cmp = DefaultDict(0, cmp)
    save_row = [params..., def_cmp[0], def_cmp[1]]
    write(file, join(save_row, ","), "\n")
    println("Time taken $telaps.")
end

# Parameters, number of extinct, number of survived.
close(file)
