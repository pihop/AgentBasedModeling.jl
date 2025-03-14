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

function build_aggregate(pitx::rxType, t) where {rxType}
    PopulationItxAggregator{typeof(pitx.itxdef.rx.method),rxType,typeof(t)}(
        pitx.itxdef.rx.method,
        IdDict{UInt, Any}(), 
        0, 
        typemax(typeof(t)), 
        typemax(typeof(t)), 
        typemax(typeof(t)), 
        typemax(typeof(t)))
end

function get_bound_functions(aggregate::PopulationItxAggregator{ExtrandeMethod{F,SpecifiedBound},rxType,S}, rx::R, tspan::T) where {F, rxType, S, R, T}
    # Bound evaluated as ratefmax at the start of tspan.
    @unpack ratefmax, Lf, ratef = rx.pitx
    return ratefmax, Lf, ratef
end

function get_bound_functions(aggregate::PopulationItxAggregator{ExtrandeMethod{F,IncreasingRate},rxType,S}, rx::R, tspan::T) where {F, rxType, S, R, T}
    # Bound evaluated as rate at the end of tspan.
    @unpack ratefmax, Lf, ratef = rx.pitx
    return ratef, Lf, ratef
end

function get_bound_functions(aggregate::PopulationItxAggregator{ExtrandeMethod{F,DecreasingRate},rxType,S}, rx::R, tspan::T) where {F, rxType, S, R, T}
    # Bound evaluated as rate at the start of tspan.
    @unpack ratefmax, Lf, ratef = rx.pitx
    return ratef, Lf, ratef
end

function get_teval(aggregate::PopulationItxAggregator{ExtrandeMethod{F,DecreasingRate},rxType,S}, rx::R, tspan::T) where {F, rxType, S, R, T}
    return tspan[1]
end

function get_teval(aggregate::PopulationItxAggregator{ExtrandeMethod{F,IncreasingRate},rxType,S}, rx::R, tspan::T) where {F, rxType, S, R, T}
    return min(tspan[1] + aggregate.Lmin, tspan[end])
end

function get_teval(aggregate::PopulationItxAggregator{ExtrandeMethod{F,SpecifiedBound},rxType,S}, rx::R, tspan::T) where {F, rxType, S, R, T}
    return tspan[1]
end

function compute_extrande_bounds!(aggregate::PopulationItxAggregator{ExtrandeMethod{F,Btype},rxType,tType},
        rxs::R, state::S, model::M, params::P, tspan::T, len::Ltype) where {rxType, tType, Btype <: Union{SpecifiedBound, IncreasingRate, DecreasingRate}, F, R, S, M, P, T, Ltype <: Int}
    rx_ = first(rxs)

    pop = state.pop_state
        
    @unpack pvec, pmod, subsrules = rx_.pitx
    ratefmax, Lf, ratef = get_bound_functions(aggregate, rx_, tspan)

    if aggregate.sampler.trait_indep 
        # If the bound is independent of trait values the same bound holds for all reactions.
        substrates = AgentState[get_agent(state, agent) for agent in rx_.substrates]

        pstate!(pmod, pvec, subsrules, model, substrates, state, tspan[1])
        aggregate.Lmin = Lf(pop, pvec_, tspan[1])

        teval = get_teval(aggregate, rx, tspan)

        pstate!(pmod, pvec, subsrules, model, substrates, state, teval)
        aggregate.Bmax = len * ratefmax(pop, pvec_, teval)
        return nothing
    end

    aggregate.Bmax = 0.0
    aggregate.Lmin = Inf

    for rx in rxs
        substrates = AgentState[get_agent(state, agent) for agent in rx.substrates]
        # Compute lookahead horizon.
        pstate!(pmod, pvec, subsrules, model, substrates, state, tspan[1])
        L = Lf(pop, pvec, tspan[1])
        if L < aggregate.Lmin
            aggregate.Lmin = L
        end

        # Get rate bound evaluated at teval. 
        teval = get_teval(aggregate, rx, tspan)
        pstate!(pmod, pvec, subsrules, model, substrates, state, teval)
        rB = ratefmax(pop, pvec, teval) 
        if rB >= 0.0 
            aggregate.Bmax += rB
        else 
            aggregate.Bmax += 0.0 
            @warn "Rate bound evaluated to $rB < 0. Small negative values can result from continuous ODE solvers overstepping.
            If large negative values check the bound functions in the model are correctly specified."
        end
    end
end

function compute_extrande_bounds!(aggregate::PopulationItxAggregator{ExtrandeMethod{F,UnknownBound},rxType,tType},
        rxs::R, state::S, model::M, params::P, tspan::T, len::Ltype) where {rxType, tType, F, R, S, M, P, T, Ltype <: Int}
    rx_ = first(rxs)

    pop = state.pop_state
    @unpack pvec, pmod, subsrules, ratefmax, Lf, ratef = first(rxs).pitx

    Bmax = 0.0
    Lmin = Inf

    for rx in rxs
        # Bound the rates for each reaction in the aggregate.
        substrates = AgentState[get_agent(state, agent) for agent in rx.substrates]
        pstate!(pmod, pvec, subsrules, model, substrates, state, tspan[1])
        Lmin_ = Lf(pop, pvec, tspan[1])

        if Lmin_ < Lmin 
            Lmin = Lmin_
        end

        # All times recorded in the simulations.
        ts = Set(reduce(vcat, [sub.simulation.t for sub in substrates]))
        B = 0.0
        for t in ts
            t < tspan[1] && continue 
            t > tspan[1] + Lmin && continue # These times outside lookahead horizon.

            pstate!(pmod, pvec, subsrules, model, substrates, state, t)
            rB = ratef(pop, pvec, t) 
            if rB > B
                B = rB
            end
        end
        Bmax += B
    end

    aggregate.Bmax = Bmax
    aggregate.Lmin = Lmin
end

function sample_(aggregate::PopulationItxAggregator{ExtrandeMethod{T,BType},rxType,S}, state, model, params, tspan; recompute=true) where {rxType,T,S, BType}
    rxs = values(aggregate.rxs)
    len = length(rxs)

    next_rx = 0
    next_rx_time = Inf

    aggregate.next_rx = next_rx
    aggregate.next_rx_time = next_rx_time

    isempty(rxs) && return nothing

    @unpack pvec, pmod, subsrules, ratefmax, Lf, ratef = first(rxs).pitx
    pop = state.pop_state

    compute_extrande_bounds!(aggregate, rxs, state, model, params, tspan, len)

    next_rx_time = min(tspan[1] + aggregate.Lmin, tspan[end])
    aggregate.Bmax <= 0 && return nothing
    prop_ttnj = tspan[1] + randexp() / aggregate.Bmax

    if prop_ttnj < next_rx_time 
        next_rx_time = prop_ttnj
        cur_rate = zero(tspan[1])
        UBmax = rand() * aggregate.Bmax

        for rx in rxs 
            substrates = AgentState[get_agent(state, agent) for agent in rx.substrates]
            pstate!(pmod, pvec, subsrules, model, substrates, state, prop_ttnj)
            r = ratef(pop, pvec, prop_ttnj)
            if r >= 0.0 
                cur_rate += r 
            else 
                cur_rate += 0.0 
                @warn "Rate evaluated to $rB < 0. Small negative values can result from continuous ODE solvers overstepping 0.
                If large negative values check the rate functions in the model are correctly specified."
            end

            if cur_rate ≥ UBmax
                aggregate.next_rx = rx.uid
                aggregate.next_rx_time = next_rx_time
                return nothing
            end
        end
    end

    aggregate.next_rx = next_rx
    aggregate.next_rx_time = next_rx_time
end

function sample_(aggregate::PopulationItxAggregator{GillespieMethod,rxType,S}, state, model, params, tspan; kwargs...) where {rxType,S}
    rxs = values(aggregate.rxs)

    next_rx = 0
    next_rx_time = Inf 

    aggregate.next_rx = next_rx 
    aggregate.next_rx_time = next_rx_time

    isempty(rxs) && return nothing

    pop = state.pop_state
    @unpack pvec, pmod, subsrules, ratef = first(rxs).pitx

    rates = zeros(length(rxs)) 
    cumsum = zeros(length(rxs)) 
    prevsum = 0.0
    idx = 1
    for rx in rxs
        substrates = AgentState[get_agent(state, agent) for agent in rx.substrates]
        # Each agent might have constants.
        pstate!(pmod, pvec, subsrules, model, substrates, state, tspan[1])
        r = ratef(pop, pvec, tspan[1])
        if r >= 0.0 
            rates[idx] = r 
            cumsum[idx] = prevsum + r
        else 
            push!(rates, 0.0)
            @warn "Rate evaluated to $r < 0. Assuming 0 but make sure rate functions are correctly specified."
        end
        prevsum = cumsum[idx]
        idx += 1
    end

    ttnj = tspan[1] + randexp() / cumsum[end] 
    njidx = findfirst(x -> x > rand()*cumsum[end], cumsum)

    if ttnj < next_rx_time
        aggregate.next_rx = collect(rxs)[njidx].uid
        aggregate.next_rx_time = ttnj
    end
end

function sample_(aggregate::PopulationItxAggregator{FirstReactionMethod,rxType,S}, state, model, params, tspan; kwargs...) where {rxType,S}
    rxs = values(aggregate.rxs)

    next_rx = 0
    next_rx_time = Inf

    aggregate.next_rx = next_rx
    aggregate.next_rx_time = next_rx_time 

    isempty(rxs) && return nothing

    pop = state.pop_state
    @unpack pvec, pmod, subsrules, ratefmax, Lf, ratef = first(rxs).pitx
    sampler = first(rxs).sampler

    for rx in rxs 
        substrates = AgentState[get_agent(state, agent) for agent in rx.substrates]
        reaction_time = sample_first_arrival(
            ratef, pop, pvec, pmod, subsrules, substrates, state, tspan, sampler, model; ratemax=ratefmax, Lf=Lf)
        reaction_time < next_rx_time && begin
            next_rx_time = reaction_time
            next_rx = rx.uid
        end
    end

    aggregate.next_rx = next_rx
    aggregate.next_rx_time = next_rx_time 
end

function sample_aggregates!(srxs::IdDict{UInt, Any}, state, model, params, tspan; recompute)
    for srx in srxs
        sample_(last(srx), state, model, params, tspan; recompute=recompute)
    end
end
