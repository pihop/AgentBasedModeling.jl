using AgentBasedModeling
using ModelingToolkit
using Catalyst
using Distributions

@independent_variables t
@variables τ(t) s(t)
@species C(t)
@parameters a μ cv L Cs Cτ

der = Differential(t)

# Age increases linearly in time t while size grows exponentially.  
@named CellDynamics = ODESystem([der(τ) ~ 1.0, der(s) ~ a*s], t)
Cell = AgentDynamics(CellDynamics, ())

# Hazard of gamma distribution.
gammahaz(μ,cv,x) = exp(logpdf(Gamma(1/cv, μ*cv), x) - logccdf(Gamma(1/cv, μ*cv), x))
# Adder rule for cell division.
γdiv(μ, cv, s, τ, a) = gammahaz(μ, cv, s - s*exp(-a*τ))
@register_symbolic γdiv(μ, cv, s, τ, a)

divide = @interaction begin
    @channel γdiv($μ, $cv, $Cs, $Cτ, a), $C --> 2*$C
    @sampler ExtrandeMethod(1/($μ * $cv), $L)
    @connections (($Cτ => $τ, $Cs => $s),)
    @transition (($τ => 0.0, $s => 0.5*$Cs), ($τ => 0.0, $s => 0.5*$Cs))
end

cell_population_model = AgentsModel([divide,], Dict(C => Cell, ))

using OrdinaryDiffEq
init_pop = repeat([C => (τ => 0.0, s => 0.1)], 1)
# SimulationParameters expects vector of parameter value pairs, simulation timespan,
# saving timestep and a solver for the agent dynamics.
ps = [a => 0.5, L => 1.0, μ => 1.0, cv => 0.1]
tspan = (0, 10.0)
Δt = 1.0
simulation_params = SimulationParameters(ps, tspan, Δt; 
    snapshot=[PopulationSnapshot(C), ])

res = simulate(cell_population_model, init_pop, simulation_params)
res.snapshot[:C]
