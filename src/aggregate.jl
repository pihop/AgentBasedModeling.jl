# Aggregate reactions of the same type.
mutable struct PopulationItxAggregator{mType,rxType,S}
    sampler::mType
    rxs::IdDict{UInt, Any}
    next_rx::UInt
    next_rx_time::S
    rate_bnd::S
    Bmax::S
    Lmin::S
end

#PopulationItx{iType, F1, F2, F3, psType, pvType, pmType, srType}
function build_aggregate(pitx::rxType, t) where {rxType}
#SimulationReaction{pType,mType,subType}
#    Tuple{sType, UInt}
    PopulationItxAggregator{typeof(pitx.itxdef.rx.method),rxType,typeof(t)}(
        pitx.itxdef.rx.method,
        IdDict{UInt, Any}(), 
        0, 
        typemax(typeof(t)), 
        typemax(typeof(t)), 
        typemax(typeof(t)), 
        typemax(typeof(t)))
end

function compute_extrande_bounds!(aggregate::A, 
        rxs::R, state::S, model::M, params::P, tspan::T, len::Ltype) where {A <: PopulationItxAggregator, R, S, M, P, T, Ltype <: Int}
    rx_ = first(rxs)
    pop_ = state.pop_state
    pvec_ = first(rxs).pitx.pvec
    pmod_ = first(rxs).pitx.pmod
    subsrules_ = first(rxs).pitx.subsrules
    Lf = first(rxs).pitx.Lf 
    ratefmax = first(rxs).pitx.ratefmax

    if aggregate.sampler.trait_indep 
        # Same bound of all reactions.
        substrates = AgentState[get_agent(state, agent) for agent in rx_.substrates]
        pstate!(pmod_, pvec_, subsrules_, model, substrates, state, tspan[1])
        aggregate.Bmax = len * ratefmax(pop_, pvec_, tspan[1])
        aggregate.Lmin = Lf(pop_, pvec_, tspan[1])
        return nothing
    end

    aggregate.Bmax = 0.0
    aggregate.Lmin = Inf
    for rx in rxs
        substrates = AgentState[get_agent(state, agent) for agent in rx.substrates]
        pstate!(pmod_, pvec_, subsrules_, model, substrates, state, tspan[1])
        L = Lf(pop_, pvec_, tspan[1]) 
        if L < aggregate.Lmin
            aggregate.Lmin = L
        end
        aggregate.Bmax += ratefmax(pop_, pvec_, tspan[1])
    end
end

function sample_(aggregate::PopulationItxAggregator{ExtrandeMethod,rxType,S}, state, model, params, tspan; recompute=true) where {rxType,S}
    rxs = values(aggregate.rxs)
    len = length(rxs)

    next_rx = 0
    next_rx_time = Inf

    aggregate.next_rx = next_rx
    aggregate.next_rx_time = next_rx_time

    isempty(rxs) && return nothing

    pop_ = state.pop_state
    pvec_ = first(rxs).pitx.pvec
    pmod_ = first(rxs).pitx.pmod
    subsrules_ = first(rxs).pitx.subsrules
    Lf = first(rxs).pitx.Lf 
    ratefmax = first(rxs).pitx.ratefmax
    ratef = first(rxs).pitx.ratef

    recompute && compute_extrande_bounds!(aggregate, rxs, state, model, params, tspan, len)

    next_rx_time = min(tspan[1] + aggregate.Lmin, tspan[end])

    prop_ttnj = tspan[1] + randexp() / aggregate.Bmax

    if prop_ttnj < next_rx_time 
        next_rx_time = prop_ttnj
        cur_rate = zero(tspan[1])
        UBmax = rand() * aggregate.Bmax

        for rx in rxs 
            substrates = AgentState[get_agent(state, agent) for agent in rx.substrates]
            pstate!(pmod_, pvec_, subsrules_, model, substrates, state, prop_ttnj)
            cur_rate += ratef(pop_, pvec_, prop_ttnj)

            if cur_rate ≥ UBmax
                aggregate.next_rx = rx.uid
                aggregate.next_rx_time = next_rx_time
                return nothing
            end
        end
    end

    aggregate.next_rx = next_rx
    aggregate.next_rx_time = next_rx_time

    return nothing
end

function sample_(aggregate::PopulationItxAggregator{GillespieMethod,rxType,S}, state, model, params, tspan; kwargs...) where {rxType,S}
    rxs = values(aggregate.rxs)

    next_rx = 0
    next_rx_time = Inf 

    aggregate.next_rx = next_rx 
    aggregate.next_rx_time = next_rx_time

    isempty(rxs) && return nothing

    pop_ = state.pop_state
    pvec_ = first(rxs).pitx.pvec
    pmod_ = first(rxs).pitx.pmod
    subsrules_ = first(rxs).pitx.subsrules
    ratef = first(rxs).pitx.ratef
    sampler = first(rxs).sampler

    total_rate = 0.0
    for rx in rxs
        substrates = AgentState[get_agent(state, agent) for agent in rx.substrates]
        # Each agent might have constants.
        pstate!(pmod_, pvec_, subsrules_, model, substrates, state, tspan[1])
        total_rate += ratef(pop_, pvec_, tspan[1])
    end

    ttnj = tspan[1] + randexp() / total_rate

#    reaction_time = sample_first_arrival(
#        total_rate, pop_, pvec_, pmod_, subsrules_, substrates, state, tspan, sampler, model; ratemax=nothing, Lf=nothing)

    aggregate.next_rx = rand(rxs).uid
    aggregate.next_rx_time = ttnj
end

function sample_(aggregate::PopulationItxAggregator{FirstReactionMethod,rxType,S}, state, model, params, tspan; kwargs...) where {rxType,S}
    rxs = values(aggregate.rxs)

    next_rx = 0
    next_rx_time = Inf

    aggregate.next_rx = next_rx
    aggregate.next_rx_time = next_rx_time 
   
    isempty(rxs) && return nothing
      
    pop_ = state.pop_state
    pvec_ = first(rxs).pitx.pvec
    pmod_ = first(rxs).pitx.pmod
    subsrules_ = first(rxs).pitx.subsrules
    ratemax = first(rxs).pitx.ratefmax
    lookahead = first(rxs).pitx.Lf
    ratefmax = first(rxs).pitx.ratefmax
    ratef = first(rxs).pitx.ratef
    sampler = first(rxs).sampler
   
    for rx in rxs 
        substrates = AgentState[get_agent(state, agent) for agent in rx.substrates]
#        substrates = rx.substrates
        reaction_time = sample_first_arrival(
            ratef, pop_, pvec_, pmod_, subsrules_, substrates, state, tspan, sampler, model; ratemax=ratefmax, Lf=lookahead)
        reaction_time < next_rx_time && begin
            next_rx_time = reaction_time
            next_rx = rx.uid
        end
    end

    aggregate.next_rx = next_rx
    aggregate.next_rx_time = next_rx_time 
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

function sample_aggregates!(srxs::IdDict{UInt, Any}, state, model, params, tspan; recompute)
    for srx in srxs
        sample_(last(srx), state, model, params, tspan; recompute=recompute)
    end
end
