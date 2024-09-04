using AgentBasedModeling
using ModelingToolkit
using Distributions
using Catalyst

# Dormancy, two types of cells.
@independent_variables t
@variables τ(t)
@species C(t) D(t)

@parameters Cτ, Dτ, T1, T2, αdiv, βdiv, wdiv, αdeath, βdeath, αendorm, βendorm, αexdorm, βexdorm, L
der = Differential(t)

# Age increases linearly in time t.  
@named CellAge = ODESystem([der(τ) ~ 1.0, ], t)
CellC = AgentDynamics((CellAge,), ())
CellD = AgentDynamics((CellAge,), ())

function gammahaz(α,β,x) 
    try 
        min(exp(logpdf(Gamma(1/β, α*β), x) - logccdf(Gamma(1/β, α*β), x)), 1/(α*β))
    catch
        return 1/(α*β)
    end
end

rate_onoff(α, β, T1, T2, τ, time) = begin
    mod(time, T1+T2) < T1 && return gammahaz(α, β, τ)
    return 0.0
end

rate_offon(α, β, T1, T2, τ, time) = begin
    mod(time, T1+T2) < T1 && return 0.0
    return gammahaz(α, β, τ)
end

bound_rate(α, β, T1, T2, τ, L, time, rate) = begin
    T = T1 + T2
    diff = 0.0 
    if (mod(time, T) <= T1) && (mod(time + L, T) >= T1)
        diff = T1 - mod(time, T) 
    elseif (mod(time, T) >= T1) && (mod(time + L, T) <= T1)
        diff = T - mod(time, T) 
    end
      
    return maximum([
        rate(α, β, T1, T2, τ, time),
        rate(α, β, T1, T2, τ + diff, time + diff),
        rate(α, β, T1, T2, τ + diff - 1e-8, time + diff - 1e-8),
        rate(α, β, T1, T2, τ + diff + 1e-8, time + diff + 1e-8),
        rate(α, β, T1, T2, τ + L, time + L)]) + 1e-4
end

hazbound(α, β, τ, L) = gammahaz(α, β, τ + L) + 1e-4

@register_symbolic rate_onoff(α, β, T1, T2, τ, time)
@register_symbolic rate_offon(α, β, T1, T2, τ, time)
@register_symbolic bound_rate(α, β, T1, T2, τ, L, time, rate::Function)
@register_symbolic gammahaz(α, β, τ)
@register_symbolic hazbound(α, β, τ, L)

enter_dormancy = @interaction begin
    @channel rate_onoff($αendorm, $βendorm, $T1, $T2, $Cτ, $t), $C --> $D
    @sampler FirstReactionMethod($bound_rate($αendorm, $βendorm, $T1, $T2, $Cτ, $L, $t, $rate_onoff), $L)
    @connections ($Cτ, $C, $τ)
    @transition (($τ => 0.0, ),)
end

exit_dormancy = @interaction begin
    @channel rate_offon($αexdorm, $βexdorm, $T1, $T2, $Dτ, $t), $D --> $C
    @sampler FirstReactionMethod($bound_rate($αexdorm, $βexdorm, $T1, $T2, $Dτ, $L, $t, $rate_offon), $L)
    @connections ($Dτ, $D, $τ)
    @transition (($τ => 0.0, ),)
end

divide = @interaction begin
    @channel gammahaz($αdiv, $βdiv, $Cτ), $C --> 2*$C
    @sampler FirstReactionMethod($hazbound($αdiv, $βdiv, $Cτ, $L), $L)
    @connections ($Cτ, $C, $τ)
    @transition ($τ => 0.0, ), ($τ => 0.0, )
end

die = @interaction begin
    @channel rate_onoff($αdeath, $βdeath, $T1, $T2, $Cτ, $t), $C --> 0 
    @sampler FirstReactionMethod($bound_rate($αdeath, $βdeath, $T1, $T2, $Cτ, $L, $t, $rate_onoff), $L)
    @transition ()
    @connections ($Cτ, $C, $τ)
end

interactions = [enter_dormancy, exit_dormancy, divide, die]
population_model = PopulationModelDef(interactions, Dict(C => CellC, D => CellD))
