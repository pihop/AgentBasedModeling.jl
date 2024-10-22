using AgentBasedModeling
using Catalyst
using Distributions
import Distributions: Categorical

# R0 = 6.8, Tlat = 5.8 days, Tinf = 14 days, ε = 1.1 × 10^3  
#
# R0 = β/(γ + μ)
# ε = μ/(γ + μ)
# γ = 1/(Tlat + Tinf)
# μ = 0.02/year

# Covid (Delta)
#R0_ = 6.8
#Tlat_ = 5.8
#Tinf_ = 14
#ε_ = 1.1 * 10^3
#μ_ = 0.02
#γ_ = 1 / (Tlat_ + Tinf_)
#β_ = R0_ / (γ_ + μ_)
# R0 = β/(γ + μ)
# ε = μ/(γ + μ)
# γ = 1/(Tlat + Tinf)

@variables t
@species I(t), S(t), R(t), τ(t)
@parameters Sτ Iτ L K n μ β γ Age LK SAge IAge a_
D = Differential(t)

@named InfectedDyn = ODESystem([D(τ) ~ 1.0, ], t)
@named SusceptibleDyn = EmptyTraitProblem() 
@named RecoveredDyn = EmptyTraitProblem()

Infected = AgentDynamics((InfectedDyn,), (Age,))
Susceptible = AgentDynamics((SusceptibleDyn,), (Age,))
Recovered = AgentDynamics((RecoveredDyn,), (Age,))

function γ_infect(τ, β, K, n, Age) 
    β_ = Age == 1 ? β / 2 : β
    return (β_ / (1 + (K / τ)^n))
end
@register_symbolic γ_infect(τ, β, K, n, Age) 
γ_recover(τ) = γ

# Two age groups 0-20, 20-
agedist = Categorical([0.24, 0.76])
function sample_age()
    rand(agedist)
end

@register_symbolic infect_bound(τ, β, K, n, Age) 

# Different versions of the infection interaction using with the two different algorithms and different definitions 
# of rate bounds.
infectionL = @interaction begin
    @channel γ_infect($Iτ, $β, $K, $n, $SAge) / ($S + $I + $R), $I + $S --> 2 * $I
    @sampler ExtrandeMethod($γ_infect($Iτ + $LK, $β, $K, $n, $SAge)/($S+$I+$R), $LK)
    @connections ($Iτ, $I, $τ) ($SAge, $S, $Age) ($IAge, $I, $Age)
    @transition (($τ => $Iτ, $Age => $IAge), ($τ => 0.0, $Age => $SAge))
    @savesubstrates ($I, $τ)
end

# Trait independent bound for the rate. 
infection_cbnd = @interaction begin
    @channel γ_infect($Iτ, $β, $K, $n, $SAge) / ($S + $I + $R), $I + $S --> 2 * $I
    @sampler ExtrandeMethod($β/($S+$I+$R), Inf; trait_indep=true)
    @connections ($Iτ, $I, $τ) ($SAge, $S, $Age) ($IAge, $I, $Age)
    @transition (($τ => $Iτ, $Age => $IAge), ($τ => 0.0, $Age => $SAge))
    @savesubstrates ($I, $τ)
end

infection_frm = @interaction begin
    @channel γ_infect($Iτ, $β, $K, $n, $SAge) / ($S + $I + $R), $I + $S --> 2 * $I
    @sampler FirstReactionMethod($γ_infect($Iτ + $LK, $β, $K, $n, $SAge)/($S+$I+$R), $LK)
    @connections ($Iτ, $I, $τ) ($SAge, $S, $Age) ($IAge, $I, $Age)
    @transition (($τ => $Iτ, $Age => $IAge), ($τ => 0.0, $Age => $SAge))
    @savesubstrates ($I, $τ)
end

# Trait independent bound for the rate. 
infection_frm_cbnd = @interaction begin
    @channel γ_infect($Iτ, $β, $K, $n, $SAge) / ($S + $I + $R), $I + $S --> 2 * $I
    @sampler FirstReactionMethod($β/($S+$I+$R), Inf; trait_indep=true)
    @connections ($Iτ, $I, $τ) ($SAge, $S, $Age) ($IAge, $I, $Age)
    @transition (($τ => $Iτ, $Age => $IAge), ($τ => 0.0, $Age => $SAge))
    @savesubstrates ($I, $τ)
end

immigration = @interaction begin
    @channel $μ*($S + $I + $R), 0 --> $S
    @variable $a_ = $sample_age() 
    @transition (($τ => 0.0, $Age => $a_), )
    @sampler ExtrandeMethod($μ*($S+$I+$R), Inf; trait_indep=true)
end

immigration_frm = @interaction begin
    @channel $μ*($S + $I + $R), 0 --> $S
    @variable $a_ = $sample_age() 
    @transition (($τ => 0.0, $Age => $a_), )
    @sampler FirstReactionMethod($μ*($S+$I+$R), Inf; trait_indep=true)
end

recovery = @interaction begin
    @channel γ_recover($Iτ), $I --> $R 
    @connections ($Iτ, $I, $τ) ($IAge, $I, $Age)
    @sampler GillespieMethod()
    @transition (($Age => $IAge, ), )
    @savesubstrates ($I, $τ)
end

s_death = @interaction begin
    @channel $μ, $S --> 0 
    @sampler GillespieMethod()
end

i_death = @interaction begin
    @channel $μ, $I --> 0 
    @sampler GillespieMethod()
end

r_death = @interaction begin
    @channel $μ, $R --> 0 
    @sampler GillespieMethod()
end

# Put them together.
behaviours = Dict(I => Infected, S => Susceptible, R => Recovered)
population_model_L = PopulationModelDef(
    [infectionL, recovery, immigration, s_death, i_death, r_death], 
    behaviours)

population_model = PopulationModelDef(
    [infection_cbnd , recovery, immigration, s_death, i_death, r_death], 
    behaviours)

population_model_frm = PopulationModelDef(
    [infection_frm_cbnd , recovery, immigration_frm, s_death, i_death, r_death], 
    behaviours)

population_model_frmL = PopulationModelDef(
    [infection_frm , recovery, immigration_frm, s_death, i_death, r_death], 
    behaviours)
