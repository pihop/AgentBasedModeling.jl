using AgentBasedModeling
using ModelingToolkit
using Catalyst
using Distributions

@variables t τ(t) s(t)
@species C(τ)
@parameters a μ cv L Cs Cτ

der = Differential(t)

@named CellDynamics = ODESystem([der(τ) ~ 1.0, der(s) ~ a*s], t)
Cell = AgentDynamics((CellDynamics,), ())
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
cell_population_model = PopulationModelDef([divide,], Dict(C => Cell, ))

using OrdinaryDiffEq
init_pop = repeat([C => (τ => 0.0, s => 0.1)], 1)
# SimulationParameters expects vector of parameter value pairs, simulation timespan,
# saving timestep and a solver for the agent dynamics.
ps = [a => 0.5, L => 1.0, μ => 1.0, cv => 0.1]
tspan = (0, 15.0)
Δt = 1.0
simulation_params = SimulationParameters(ps, tspan, Δt, Tsit5();
    snapshot=[PopulationSnapshot(C), ])
res = simulate(cell_population_model, init_pop, simulation_params)
res.snapshot[:C]

