# AgentBasedModeling.jl -- a tool for stochastic simulation of structured population dynamics.
This package for [Julia](http://www.julialang.org) provides a tool for easy specification and simulation of stochastic
agent-based models where internal continuous-time agent state dynamics explicitly modelled and affect the population
level interactions between agents.

* Individual agent dynamics defined as ODEs, SDEs or jump-diffusion processes using [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) and [Catalyst.jl](https://github.com/SciML/Catalyst.jl).
* High-level symbolic specification of agent interaction rules.
* Stochastic sampling for rate-based interactions between agents. 

## Installation
The package can be installed through the Julia package manager.

```julia
pkg> add https://github.com/pihop/AgentBasedModeling.jl.git
```

## Get started
Example providing cell population dynamics.
Start by defining the continuous time dynamics of the agents.

```julia
using AgentBasedModeling
using ModelingToolkit
using Catalyst
using Distributions

@variables t τ(t) s(t)
@species C(τ)
@parameters a μ cv L Cs Cτ

der = Differential(t)

# Age increases linearly in time t while size grows exponentially.  
@named CellDynamics = ODESystem([der(τ) ~ 1.0, der(s) ~ a*s], t)
Cell = AgentDynamics((CellDynamics,), ())
```

Define the division interaction dividing the cell and creating two daughter cells with half the size each. 
```julia
# Hazard of gamma distribution.
gammahaz(μ,cv,x) = exp(logpdf(Gamma(1/cv, μ*cv), x) - logccdf(Gamma(1/μ, μ*cv), x))
# Adder rule for cell division.
γdiv(μ, cv, s, τ, a) = gammahaz(μ, cv, s - s*exp(-a*τ))
@register_symbolic γdiv(μ, cv, s, τ, a)

divide = @interaction begin
    @channel γdiv($μ, $cv, $Cs, $Cτ, a), $C --> 2*$C
    @sampler ExtrandeMethod(1/($μ * $cv), $L)
    @connections ($Cτ, $C, $τ) ($Cs, $C, $s)
    @transition ($τ => 0.0, $s => 0.5*$Cs), ($τ => 0.0, $s => 0.5*$Cs)
end
```
Combine the agent dynamics and interactions into a model.
```julia
cell_population_model = PopulationModelDef([divide,], Dict(C => Cell, ))
```
Give some initial values and specify the model parameters. Population is initialised with a single cell with age 0 and size 0.1.
```julia
using OrdinaryDiffEq
init_pop = repeat([C => (τ => 0.0, s => 0.1)], 1)
# SimulationParameters expects vector of parameter value pairs, simulation timespan,
# saving timestep and a solver for the agent dynamics.
ps = [a => 0.5, L => 1.0, μ => 1.0, cv => 0.1]
tspan = (0, 10.0)
Δt = 1.0
simulation_params = SimulationParameters(ps, tspan, Δt, Tsit5();
    snapshot=[PopulationSnapshot(C), ])
```
Simulate the system and show display the population size snapshots.
```julia
res = simulate(cell_population_model, init_pop, simulation_params)
res.snapshot[:C]
```

The package is flexible --- see the following for more complex examples!
* SIR model with incubation periods.
* Cell population with gene expression coupled to growth and division.
