mutable struct SimulationState{N,idType}
    t::Float64
    pop::Dict{Num, Dict{idType, AgentState}}
    pop_state::NTuple{N, Int64}
    srxs::Dict{PopulationItx, PopulationItxAggregator}

    function SimulationState(t, pop, rxs)
        state = new{length(pop),idType}()
        state.t = t
        state.pop = pop
        state.srxs = Dict{PopulationItx, PopulationItxAggregator}(
            rx => build_aggregate(rx, t) for rx in rxs)
        return state
    end
end

function update_sampler!(sitx::SimulationReaction, pop_state, tspan)
    update_sampler!(sitx.sampler, pop_state, sitx.pstate, tspan)
end

function update_sampler!(pitxa::PopulationItxAggregator, state::SimulationState, tspan)
    update_sampler!.(pitxa.rxs, Ref(state.pop_state), Ref(tspan))
end

function update_sampler!(aggregates::Pair{PopulationItx,PopulationItxAggregator}, state::SimulationState, tspan)
    update_sampler!(aggregates[2], state, tspan)
end

function update_sampler!(state::SimulationState, tspan)
    for srx in state.srxs
        update_sampler!.(srx, Ref(state), Ref(tspan))
    end
end

struct SimulationParameters{T,DEAlg,JAgg,K,S}
    ps::T
    tspan::Tuple{Float64, Float64}
    Δt::Float64
    solver::DEAlg
    jumpaggregator::JAgg
    solverkws::K
    snapshot::S
    jitt::Float64
    maxpop::Float64

    function SimulationParameters(ps::T, tspan, Δt, solver::DEAlg; jitt=1e-4, maxpop=Inf, snapshot::S=[], jumpaggregator::JAgg=Direct(), solverkws::K=()) where {DEAlg<:SciMLBase.DEAlgorithm, JAgg<:JumpProcesses.AbstractAggregatorAlgorithm, T,K,S}
        new{T,DEAlg,JAgg,K,S}(ps, tspan, Δt, solver, jumpaggregator, solverkws, snapshot, jitt, maxpop)
    end
end

function substitute_agent(subs, agents, pop, model)
    isempty(subs) && return ()

    idx = findfirst(x -> x in keys(agents), subs)
    isnothing(idx) && return ()

    subs[idx] = values(agents[subs[idx]])
    idxs_ = Iterators.flatten((max(1, idx-1):idx-1, idx+1:length(subs)))
       
    for idx_ in idxs_
        subs[idx_] = values(pop[subs[idx_]])
    end
      
    return Iterators.product(subs...)
end

function make_reactions!(agents, state, model::PopulationModel, tspan, params; make_zero_substrate_rx=true)
    # Construct pairs of agents that can take part in a reaction.
    # Make a dict of agents => rn_sym.
    for rx in model.rxs
        method = rx.itxdef.rx.method
        substrates = rx.itxdef.rx.rx.substrates
        substoich = rx.itxdef.rx.rx.substoich

        isempty(substrates) && !make_zero_substrate_rx && continue
        
        subs::Vector{Any} = vcat(fill.(Num.(substrates), substoich)...)
        reacts = substitute_agent(subs, agents, state.pop, model)

        isempty(substrates) && begin 
            srx = SimulationReaction(rx, (), method, model.rn)
            state.srxs[rx].rxs[srx.uid] = srx 
            continue
        end

        for react in reacts
            srx = SimulationReaction(rx, react, method, model.rn)
            for r in react
                push!(r.srxs, (rx, srx.uid))
            end
            state.srxs[rx].rxs[srx.uid] = srx 
        end 
    end
end

function simulate_internal(problem, agent, init, tspan, ps, solver; model, kwargs...)
#    pnew = Tuple(Symbolics.unwrap.(substitute(p, Dict(ps...))) for p in parameters(model.traitdefs[agent.sym].dynamics))

    u0 = [Symbolics.unwrap.(substitute(p, Dict(init...))) for p in unknowns(model.traitdefs[agent.sym].dynamics)]
    if !isempty(ps) 
        prob = remake(problem, u0=u0, tspan=tspan)
        return solve(prob, solver; kwargs...) 
    else
        prob = remake(problem, u0=u0, tspan=tspan)
        return solve(prob, solver; kwargs...) 
    end
end

function append_sim!(problem, agent, agentsim::Nothing, tspan, ps, solver; model)
    init = agent.init_trait
    sim = simulate_internal(
        problem, agent, init, (agent.btime, tspan[end]), ps, solver; model=model)

    agent.simulation = sim
    Interpolations.deduplicate_knots!(agent.simulation.t)
    agent.simulation_interp = interpolate((agent.simulation.t, ), agent.simulation.u, Gridded(Linear()))
end

function append_sim!(::EmptyTraitProblem, agent, agentsim::Nothing, tspan, ps, solver; model)
    nothing
end

function append_sim!(problem, agent, agentsim::Union{ODESolution, RODESolution}, tspan, ps, solver; model)
    k_ = first.(agent.init_trait)
    init = Tuple(k_ .=> agent.simulation(tspan[1]; idxs=collect(k_)))

    sim = simulate_internal(problem, agent, init, tspan, ps, solver; model=model)

    agent.simulation = SciMLBase.build_solution(
        sim.prob, 
        :NoAlgorithm,
        [agentsim.t; sim.t], 
        [agentsim.u; sim.u], 
        successful_retcode=true)
    Interpolations.deduplicate_knots!(agent.simulation.t; move_knots = true)
    agent.simulation_interp = interpolate((agent.simulation.t, ), agent.simulation.u, Gridded(Linear()))
end

function simulate_traits!(pop, tstart, tend, params; model, kwargs...)
    for (uid,agent) in Iterators.flatten(values(pop))
        append_sim!(
            model.traitprobs[agent.sym].problem, agent, agent.simulation, (tstart, tend), params.ps, params.solver; model=model)
    end
end

function population_state_vector(pop, rn)
    return Tuple(pop[s] for s in unknowns(rn))
end

struct AmbigiousConnection <: Exception end
struct TraitDefinitionMissing <: Exception end
struct BirthDefinitionMissing <: Exception end

function update_pop_state!(state::SimulationState, model::PopulationModel)
    pop_state = Dict(k => length(state.pop[k]) for k in keys(state.pop);) 
    state.pop_state = population_state_vector(pop_state, model.rn) 
end

function compute_new_agents(srx, state, time, model::PopulationModel, params::SimulationParameters; kwargs...) 
    new = Dict{Num, Dict{idType, AgentState}}()

    products = srx.pitx.itxdef.rx.rx.products
    prodstoich = srx.pitx.itxdef.rx.rx.prodstoich

    # Construct input.
     
    new_agents = vcat(fill.(products, prodstoich)...)
    new_traits = trait_transition(srx.pitx, new_agents, srx.substrates, srx.pitx.subsrules, state, model, time)

    pstate!(srx.pitx.pmod, srx.pitx.pvec, srx.pitx.subsrules, model, srx.substrates, state, time)
    varsubs = variable_subs(srx.pitx.itxdef.vars, srx.pitx.pvec, srx.pitx.psymbs)

    for (i, agent) in enumerate(new_agents)
        dyn = model.traitdefs[agent].dynamics
        cts = model.traitdefs[agent].constants      

        !in(agent, keys(new)) && begin new[agent] = Dict{idType, AgentState}() end
        
        isempty(new_traits) && begin
            # Early return for the agents with no traits.
            agent_ = AgentState(time, agent, (), (), [(s.sym, s.uid) for s in srx.substrates])
            new[agent][agent_.uid] = agent_
            continue
        end

        alltraits_ = Tuple(t[1] => Symbolics.unwrap.(substitute(t[2], varsubs)) for t in new_traits[i])
        tr = Tuple(x => Symbolics.unwrap.(substitute([Num(x), ], alltraits_)...) for x in unknowns(dyn))
        c = Tuple(x => Symbolics.unwrap.(substitute([Num(x), ], alltraits_)...) for x in cts)
        agent_ = AgentState(time, agent, tr, c, [(s.sym, s.uid) for s in srx.substrates])
        new[agent][agent_.uid] = agent_
    end
    return new, srx.substrates
end

function push_to_pop!(pop::Dict, agents::Dict) 
    for ksym in keys(agents)
        !in(ksym, keys(pop)) && begin pop[ksym] = Dict{idType, AgentState}() end
        for kint in keys(agents[ksym])
            pop[ksym][kint] = agents[ksym][kint]
        end
    end
end

function push_to_pop!(pop::Dict, agents::Tuple) 
    for agent in agents
        !in(agent.sym, keys(pop)) && begin pop[agent.sym] = Dict{idType, AgentState}() end
        pop[agent.sym][agent.uid] = agent 
    end
end

function filter_agent_pop!(state::SimulationState, delagents)
    for del in delagents
        pop!(state.pop[del[1]], del[2])
    end
end

function get_srxs(state, agent)
    agent = get(state.pop[agent[1]], agent[2], nothing)
    isnothing(agent) && return nothing
    return agent.srxs
end

function get_substrates(state, rx)
    srx = get(state.srxs[rx[1]].rxs, rx[2], nothing)
    isnothing(srx) && return nothing
    return srx.substrates
end

function remove_agent!(state::SimulationState, agent)
    agent_ = pop!(state.pop[agent.sym], agent.uid, nothing)
    isnothing(agent_) && return nothing
    remove_reaction!(state, agent_)
end

function remove_reaction!(state::SimulationState, agent::AgentState)
    isempty(agent.srxs) && return nothing
    for srx in agent.srxs
        remove_reaction!(state, srx)  
    end
end

function remove_reaction!(state::SimulationState, rx)
    srx = pop!(state.srxs[rx[1]].rxs, rx[2], nothing)
    return nothing
    isnothing(srx) && return nothing
end

function filter_rxs!(state::SimulationState, delagents)
    isempty(delagents) && return nothing
    for agent in delagents
        remove_agent!(state, agent)    
    end
end

function update_dtime!(time, deleted, agents)
    for agent in deleted
        setfield!(agent, :dtime, time)
    end
end

function log_products!(srx::SimulationReaction, state, rxtime, agents, model, results::SimulationResults)
    saving = srx.pitx.itxdef.saving
    prod_traits = filter(x -> x isa SaveProductTrait, saving) 
    isempty(prod_traits) && return nothing 

    for agent in Iterators.flatten(values.(values(agents)))
        for save in prod_traits 
            name = save_trait_name(save)
            !in(name, keys(results.prods)) && begin results.prods[agent.sym] = [] end

            idx = indexof(save.trait, unknowns(model.traitdefs[agent.sym].dynamics))

            !isnothing(idx) && begin
                push!(results.prods[name], 
                    TraitValueRx(agent.init_trait[idx][1], agent.init_trait[idx][2], rxtime, agent.idx, srx.pitx.itxdef))
                continue
            end
        end
    end
end

function log_substrates!(srx::SimulationReaction, state, rxtime, agents, model, results::SimulationResults)
    saving = srx.pitx.itxdef.saving
    subs_traits = filter(x -> x isa SaveSubstrateTrait, saving) 
    isempty(saving) && return nothing 

    for agent in agents
        for save in subs_traits 
            name = save_trait_name(save)
            !in(name, keys(results.subs)) && begin results.subs[agent] = [] end

            idx = indexof(save.trait, unknowns(model.traitdefs[agent.sym].dynamics))

            !isnothing(idx) && begin
                push!(results.subs[name], 
                    TraitValueRx(agent.init_trait[idx][1], agent.simulation(rxtime)[idx], rxtime, agent.idx, srx.pitx.itxdef))
                continue
            end
        end
    end
end

function log_snapshot!(time, saving, state::SimulationState, model, results::SimulationResults)
    isempty(saving) && return nothing

    for save in saving 
        snapshot = TraitValue[]
        snapshot_n = [] 
        name = save_trait_name(save) 

        for agent in Iterators.flatten(values.(values(state.pop)))
            if save isa TraitSnapshot 
                isa(model.traitdefs[agent.sym].dynamics, EmptyTraitProblem) && continue
                push!(snapshot, TraitValue(agent.simulation(time; idxs=save.trait)[1], time, agent.idx))
            elseif save isa PopulationSnapshot
                isequal(agent.sym, save.agent) ? push!(snapshot_n, agent.sym) : nothing
            end 
        end
        
        if save isa TraitSnapshot
            push!(results.snapshot[name], Snapshot(time, snapshot))
        elseif save isa PopulationSnapshot
            push!(results.snapshot[name], Snapshot(time, length(snapshot_n)))
        end
    end
end

function initialise_agents(model, init_pop, tspan, params::SimulationParameters; kwargs...) 
    # Make the population state dictionary.
    pop = Dict{Num, Dict{idType, AgentState}}(
        Num(s) => Dict{idType, AgentState}() for s in unknowns(model.rn))
    for (agent, init_traits) in init_pop 
        dyn = model.traitdefs[agent].dynamics
        cts = model.traitdefs[agent].constants

        c = Tuple(x => Symbolics.unwrap.(substitute([x, ], init_traits)...) for x in cts)
        tr = Tuple(x => Symbolics.unwrap.(substitute([Num(x), ], init_traits)...) for x in unknowns(dyn))
      
        agent_ = AgentState(tspan[1], agent, tr, c, nothing)
        pop[agent][agent_.uid] = agent_ 
    end
    return pop
end

function init_simulator(modeldef, init_pop, params)
    model = PopulationModel(modeldef, params) 

    population = initialise_agents(model, init_pop, (params.tspan[1], params.tspan[1] + params.Δt), params)
    state = SimulationState(params.tspan[1], population, model.rxs)
    results = SimulationResults(modeldef; snapshot=params.snapshot)
    update_pop_state!(state, model)

    simulate_traits!(state.pop, state.t, state.t + params.Δt, params; model=model)
    make_reactions!(state.pop, state, model, (state.t, state.t + params.Δt), params)

    return state, results, model
end

function simulate(modeldef::PopulationModelDef, init_pop, params::SimulationParameters; 
    showprogress=true, 
    save_interactions=false,
    remember_all_agents=false) 

    state, results, model = init_simulator(modeldef, init_pop, params)
    
    progress = ProgressUnknown()

    all_agents = Dict{Num, Dict{idType, AgentState}}()
    log_snapshot!(state.t, params.snapshot, state, model, results)
    tend = minimum([state.t + params.Δt, params.tspan[end]])
    recompute_bounds = true

    try 
        while true
            sample_aggregates!(state.srxs, state, model, params, (state.t, tend), recompute=recompute_bounds)
            next_rx_time, rx_channel = findmin(x -> x.next_rx_time, state.srxs)
            rxidx = state.srxs[rx_channel].next_rx 

            if next_rx_time < tend && rxidx != 0
                srx = state.srxs[rx_channel].rxs[rxidx]
                new_agents, deleted_agents = compute_new_agents(srx, state, next_rx_time, model, params)
                update_dtime!(next_rx_time, deleted_agents, state.pop)
                
                # Logging
                log_substrates!(srx, state, next_rx_time, deleted_agents, model, results)

                # Remove agents involved in the current reaction and reactions with
                # them as substrates.
                filter_rxs!(state, deleted_agents)

                # Simulate traits of the new agents to the end of the tspan.   
                simulate_traits!(new_agents, next_rx_time, tend, params; model=model)

                log_products!(srx, state, next_rx_time, new_agents, model, results)

                # Add the new to the population state.
                remember_all_agents && push_to_pop!(all_agents, deleted_agents)
                push_to_pop!(state.pop, new_agents)

                # New reactions.
                make_reactions!(new_agents, state, model, (next_rx_time, tend), params; make_zero_substrate_rx=false)
                state.t = next_rx_time 
                save_interactions && push!(results.interactions, (next_rx_time, srx, state.pop))
                update_pop_state!(state, model)
            elseif next_rx_time < tend && rxidx == 0 
                state.t = next_rx_time 
            else 
                state.t = tend 
                tend = minimum([state.t + params.Δt, params.tspan[end]])
                simulate_traits!(state.pop, state.t, tend, params; model=model)
                log_snapshot!(state.t, params.snapshot, state, model, results)
            end

            pop_size = length(collect(Iterators.flatten(values.(values(state.pop)))))
            state.t ≥ params.tspan[end] && break
            pop_size ≥ params.maxpop && break
            
            showprogress && ProgressMeter.next!(progress, showvalues = [("Time", state.t), ("Populations size", pop_size)])
        end
    catch e
        if e isa InterruptException
            println("Simulation interrupted! Saving results.")
            log_snapshot!(state.t, params.snapshot, state, model, results)
            results.tend = state.t
      
            remember_all_agents && push_to_pop!(all_agents, state.pop)
            results.agents = all_agents 
            results.final_pop = state.pop
            return results
        else
            rethrow(e)
        end
    end

    log_snapshot!(state.t, params.snapshot, state, model, results)
    results.tend = state.t

    remember_all_agents && push_to_pop!(all_agents, state.pop)
    results.agents = all_agents 
    results.final_pop = state.pop
    return results
end
