# Aggregate reactions of the same type.
mutable struct PopulationItxAggregator{mType,N1,cType,sType,F1,F2,N2,S}
    sampler::mType
    rxs::Dict{UInt, SimulationReaction{mType,N1,cType,sType,F1,F2,N2}}
    next_rx::UInt
    next_rx_time::S
    rate_bnd::S
    Bmax::S
    Lmin::S
end

function build_aggregate(pitx::PopulationItx{mType,N1,cType,sType,F1,F2,N2}, t) where 
    {mType,N1,cType,sType,F1,F2,N2}

    PopulationItxAggregator{mType,N1,cType,sType,F1,F2,N2,typeof(t)}(
        pitx.itxdef.rx.method,
        Dict{UInt, SimulationReaction{mType,N1,cType,sType,F1,F2,N2}}(), 
        0, 
        typemax(typeof(t)), 
        typemax(typeof(t)), 
        typemax(typeof(t)), 
        typemax(typeof(t)))
end

function compute_extrande_bounds!(aggregate, rxs, state, model, params, tspan, len)
    pop_ = state.pop_state
    pvec_ = first(rxs).pitx.pvec
    pmod_ = first(rxs).pitx.pmod
    subsrules_ = first(rxs).pitx.subsrules

    if aggregate.sampler.trait_indep 
        # Same bound of all reactions.
        rx = first(rxs)
        substrates = rx.substrates
        pstate!(pmod_, pvec_, subsrules_, model, substrates, state, tspan[1])
        aggregate.Bmax = length(rxs) * rx.pitx.ratefmax(pop_, pvec_, tspan[1])
        aggregate.Lmin = rx.pitx.Lf(pop_, pvec_, tspan[1])
        return nothing
    end

    aggregate.Bmax = 0.0
    aggregate.Lmin = Inf
    for rx in rxs
        substrates = rx.substrates
        pstate!(pmod_, pvec_, subsrules_, model, substrates, state, tspan[1])
        L = rx.pitx.Lf(pop_, pvec_, tspan[1]) 
        if L < aggregate.Lmin
            aggregate.Lmin = L          
        end
#        display(rx.pitx.ratefmax)
        aggregate.Bmax += rx.pitx.ratefmax(pop_, pvec_, tspan[1])
    end
end

function sample_(aggregate::PopulationItxAggregator{ExtrandeMethod,N1,cType,sType,F1,F2,N2,S}, state, model, params, tspan; recompute=true) where {N1,cType,sType,F1,F2,N2,S}
    aggregate.next_rx = 0
    rxs = values(aggregate.rxs)
    len = length(rxs)
    aggregate.next_rx_time = Inf       

    isempty(rxs) && return nothing

    pop_ = state.pop_state
    pvec_ = first(rxs).pitx.pvec
    pmod_ = first(rxs).pitx.pmod
    subsrules_ = first(rxs).pitx.subsrules

    recompute && compute_extrande_bounds!(aggregate, rxs, state, model, params, tspan, len)

    aggregate.next_rx_time = min(tspan[1] + aggregate.Lmin, tspan[end])
    
    prop_ttnj = tspan[1] + randexp() / aggregate.Bmax

    if prop_ttnj < aggregate.next_rx_time 
        aggregate.next_rx_time = prop_ttnj
        cur_rate = zero(tspan[1])
        UBmax = rand(Uniform(0, 1)) * aggregate.Bmax

        for rx in rxs 
            pstate!(pmod_, pvec_, subsrules_, model, rx.substrates, state, prop_ttnj)
            cur_rate += rx.pitx.ratef(pop_, pvec_, prop_ttnj)

            if cur_rate ≥ UBmax
                aggregate.next_rx = rx.uid
                return nothing
            end
        end
    end

    return nothing
end

function sample_(aggregate::PopulationItxAggregator{GillespieMethod,N1,cType,sType,F1,F2,N2,S}, state, model, params, tspan; recompute=true) where {N1,cType,sType,F1,F2,N2,S}
    rxs = values(aggregate.rxs)

    aggregate.next_rx = 0
    aggregate.next_rx_time = Inf

    isempty(rxs) && return nothing

    rx = first(rxs)
    substrates = rx.substrates

    pop_ = state.pop_state
    pvec_ = rx.pitx.pvec
    pmod_ = rx.pitx.pmod
    subsrules_ = rx.pitx.subsrules

    pstate!(pmod_, pvec_, subsrules_, model, substrates, state, tspan[1])
    
    sum_rate = length(rxs) * rx.pitx.ratef(pop_, pvec_, tspan[1])
    aggregate.next_rx_time = tspan[1] + randexp() / sum_rate
    aggregate.next_rx = rx.uid
end

#function sample_(aggregate::PopulationItxAggregator, state, model, params, tspan, method::FirstReactionMethod; kwargs...) 
#    rxs = values(aggregate.rxs)
#
#    aggregate.next_rx = 0
#    aggregate.next_rx_time = tspan[end]
#    
#    isempty(rxs) && return nothing
#      
#    pop_ = state.pop_state
#    pvec_ = first(rxs).pitx.pvec
#    pmod_ = first(rxs).pitx.pmod
#    subsrules_ = first(rxs).pitx.subsrules
#
#    for rx in rxs 
#        ratemax = rx.pitx.ratefmax
#        substrates = rx.substrates
#        reaction_time = sample_first_arrival(
#            rx.pitx.ratef, pop_, pvec_, pmod_, subsrules_, substrates, state, tspan, rx.sampler, model; ratemax=ratemax)
#        reaction_time < aggregate.next_rx_time && begin
#            aggregate.next_rx = rx.uid
#            aggregate.next_rx_time = reaction_time
#        end
#    end
#end
#
function sample_(aggregate::PopulationItxAggregator{FirstReactionMethod ,N1,cType,sType,F1,F2,N2,S}, state, model, params, tspan; kwargs...) where {N1,cType,sType,F1,F2,N2,S}
    rxs = values(aggregate.rxs)

    aggregate.next_rx = 0
    aggregate.next_rx_time = Inf 
    
    isempty(rxs) && return nothing
      
    pop_ = state.pop_state
    pvec_ = first(rxs).pitx.pvec
    pmod_ = first(rxs).pitx.pmod
    subsrules_ = first(rxs).pitx.subsrules

    for rx in rxs 
        ratemax = rx.pitx.ratefmax
        substrates = rx.substrates
        lookahead = rx.pitx.Lf
        reaction_time = sample_first_arrival(
            rx.pitx.ratef, pop_, pvec_, pmod_, subsrules_, substrates, state, tspan, rx.sampler, model; ratemax=ratemax, Lf=lookahead)
        reaction_time <= aggregate.next_rx_time && begin
            aggregate.next_rx_time = reaction_time
            aggregate.next_rx = rx.uid
        end
    end
end

#function sample_(aggregate::PopulationItxAggregator{DirectSampler,N1,cType,sType,F1,F2,N2,S}, state, model, params, tspan, method::FirstReactionMethod; kwargs...) where {N1,cType,sType,F1,F2,N2,S}
#    rxs = values(aggregate.rxs)
#
#    aggregate.next_rx_time = Inf #tspan[end] 
#    aggregate.next_rx = 0
#    isempty(rxs) && return nothing
#      
#    pop_ = state.pop_state
#    pvec_ = first(rxs).pitx.pvec
#    pmod_ = first(rxs).pitx.pmod
#    subsrules_ = first(rxs).pitx.subsrules
#
#    aggregate.next_rx_time = min(tspan[1] + first(rxs).pitx.Lf(pop_, pvec_, tspan[1]), tspan[end])
#
#    for rx in rxs 
#        ratemax = rx.pitx.ratefmax
#        Lf = rx.pitx.Lf
#
#        substrates = rx.substrates
#        reaction_time = sample_first_arrival(
#            rx.pitx.ratef, pop_, pvec_, pmod_, subsrules_, substrates, state, tspan, rx.sampler, model; ratemax=ratemax, Lf=Lf)
#        reaction_time <= aggregate.next_rx_time && begin
#            aggregate.next_rx = rx.uid
#            aggregate.next_rx_time = reaction_time 
#        end
#    end
#end

function sample_aggregates!(srxs::Dict{PopulationItx, PopulationItxAggregator}, state, model, params, tspan; recompute)
    for srx in srxs
        sample_(last(srx), state, model, params, tspan; recompute=recompute)
    end
end
