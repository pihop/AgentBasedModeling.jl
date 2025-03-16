mutable struct SimulationState{N,idType}
    t::Float64
    pop::Dict{Num, IdDict{idType, Any}}
    pop_state::NTuple{N, Int64}
    srxs::IdDict{idType, Any}

    function SimulationState(t, pop, rxs)
        state = new{length(pop),idType}()
        state.t = t
        state.pop = pop
        state.srxs = IdDict{idType, Any}(
            rx.uid => build_aggregate(rx, t) for rx in rxs)
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

struct SimulationParameters{T,DEAlg,JumpAlg,JAgg,K,S,iType}
    ps::T
    tspan::Tuple{Float64, Float64}
    Δt::Float64
    solver::DEAlg
    jumpsolver::JumpAlg
    jumpaggregator::JAgg
    solverkws::K
    snapshot::S
    jitt::Float64
    maxpop::Float64
    interpolation::iType

    function SimulationParameters(
        ps::T, 
        tspan, 
        Δt, 
        solver::DEAlg=Rodas4(); 
        jitt=1e-4, 
        maxpop=Inf, 
        snapshot::S=[], 
        jumpsolver=SSAStepper(), 
        jumpaggregator::JAgg=Direct(), 
        interpolation=SciMLBase.LinearInterpolation,
        solverkws::K=()) where {DEAlg, JAgg<:JumpProcesses.AbstractAggregatorAlgorithm, T,K,S}

        new{T,DEAlg,typeof(jumpsolver),JAgg,K,S,typeof(interpolation)}(
            ps, tspan, Δt, solver, jumpsolver, jumpaggregator, solverkws, snapshot, jitt, maxpop, interpolation)
    end
end

function substitute_agent(subs, agents, pop, model)
    isempty(subs) && return ()

    substitutions = []

    for agent in agents
        idxs = findall(x -> isequal(x, agent[1]), subs)
        isnothing(idxs) && continue

        subbed = Vector{Base.ValueIterator{IdDict{UInt64, Any}}}(undef, length(subs))

        for idx in idxs
            subbed[idx] = values(agents[subs[idx]])
            idxs_ = Iterators.flatten((max(1, idx-1):idx-1, idx+1:length(subs)))
            for idx_ in idxs_
                subbed[idx_] = values(pop[subs[idx_]])
            end

            # Make sure combinations with duplicate agents are removed.
            push!(substitutions, Iterators.filter(allunique, Iterators.product(subbed...)))
        end
    end
    return Iterators.flatten(substitutions)
end

function make_reactions!(agents::aType, state::sType, model::mType, tspan::tType, params::pType; make_zero_substrate_rx=true) where {aType, sType, mType, tType, pType}
    # Construct collections of agents that can take part in a reaction.
    # Make a dict of agents => rn_sym.
    for rx in model.rxs
        method = rx.itxdef.rx.method
        substrates = rx.itxdef.rx.rx.substrates
        substoich = rx.itxdef.rx.rx.substoich

        isempty(substrates) && !make_zero_substrate_rx && continue

        subs = vcat(fill.(Num.(substrates), substoich)...)
        reacts = substitute_agent(subs, agents, state.pop, model)

        isempty(substrates) && begin 
            srx = SimulationReaction(rx, (), method)
            state.srxs[rx.uid].rxs[srx.uid] = srx 
            continue
        end

        for react in reacts
            srx = SimulationReaction(rx, react, method)
            state.srxs[rx.uid].rxs[srx.uid] = srx 
        end 
    end
end

function simulate_internal(problem, agent, init, tspan, params; model, kwargs...)
    prob = remake(problem, u0=init, tspan=tspan)
    if (problem isa JumpProblem && problem.prob isa DiscreteProblem)
        return solve(prob, params.jumpsolver; kwargs...), params.jumpsolver 
    else 
        return solve(prob, params.solver; kwargs...), params.solver
    end
end

function append_sim!(problem, agent, agentsim::Nothing, tspan, params; model)
    init = agent.init_trait

    sim, alg = simulate_internal(
        problem, agent, init, (agent.btime, tspan[end]), params; model=model)

    agent.simulation = SciMLBase.build_solution(
        sim.prob, 
        alg,
        sim.t, 
        sim.u, 
        successful_retcode=true,
        interp = params.interpolation(sim.t, sim.u)
       )
end

function append_sim!(::EmptyTraitProblem, agent, agentsim::Nothing, tspan, params; model)
    nothing
end

function append_sim!(problem, agent, agentsim::Union{ODESolution, RODESolution}, tspan, params; model)
    k_ = first.(agent.init_trait)
    init = Tuple(k_ .=> agent.simulation(tspan[1]; idxs=collect(k_)))

    tspan[2] == tspan[1] && return nothing
    sim, alg = simulate_internal(problem, agent, init, tspan, params; model=model)
    ts = [agentsim.t; sim.t]
    Interpolations.deduplicate_knots!(ts)
    agent.simulation = SciMLBase.build_solution(
        sim.prob, 
        alg,
        ts, 
        [agentsim.u; sim.u], 
        successful_retcode=true,
        interp = params.interpolation(ts, [agentsim.u; sim.u])
       )
end

function simulate_traits!(pop, tstart, tend, params; model, kwargs...)
    for (uid,agent) in Iterators.flatten(values(pop))
        append_sim!(
            model.traitprobs[agent.sym].problem, agent, agent.simulation, (tstart, tend), params; model=model)
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

function compute_new_agents(srx::srxType, state::sType, time::tType, model::PopulationModel, params::SimulationParameters) where {srxType, sType, tType}
    new = Dict{Num, IdDict{idType, Any}}()

    products = srx.pitx.itxdef.rx.rx.products
    prodstoich = srx.pitx.itxdef.rx.rx.prodstoich
    substrates = [get_agent(state, agent) for agent in srx.substrates]

    # Construct input.

    new_agents = vcat(fill.(products, prodstoich)...)
    new_traits = trait_transition(srx.pitx, new_agents, substrates, srx.pitx.subsrules, state, model, time)

    pstate!(srx.pitx.pmod, srx.pitx.pvec, srx.pitx.subsrules, model, substrates, state, time)
    varsubs = variable_subs(srx.pitx.itxdef.vars, srx.pitx.pvec, srx.pitx.psymbs)

    for (i, agent) in enumerate(new_agents)
        dyn = model.traitdefs[agent].dynamics
        cts = model.traitdefs[agent].constants

        !in(agent, keys(new)) && begin new[agent] = IdDict{idType, Any}() end

        isempty(new_traits) && begin
            # Early return for the agents with no traits.
            agent_ = AgentState(time, agent, (), (), Vector{Tuple{Num, idType}}(getsymid.(substrates)), srx.uid)
            new[agent][agent_.uid] = agent_
            continue
        end
        alltraits_ = Tuple(t[1] => Symbolics.unwrap.(substitute(t[2], varsubs)) for t in new_traits[i])
        tr = Tuple(x => Symbolics.unwrap.(substitute([Num(x), ], alltraits_)...) for x in unknowns(dyn))
        c = Tuple(x => Symbolics.unwrap.(substitute([Num(x), ], alltraits_)...) for x in cts)
        agent_ = AgentState(time, agent, tr, c, Vector{Tuple{Num, idType}}(getsymid.(substrates)), srx.uid)
        new[agent][agent_.uid] = agent_
    end
    return new, substrates
end

function push_to_pop!(pop::Dict, agents::Dict) 
    for ksym in keys(agents)
        !in(ksym, keys(pop)) && begin pop[ksym] = IdDict{idType, Any}() end
        for kint in keys(agents[ksym])
            pop[ksym][kint] = agents[ksym][kint]
        end
    end
end

function push_to_pop!(pop::Dict, agents) 
    for agent in agents
        !in(agent.sym, keys(pop)) && begin pop[agent.sym] = IdDict{idType, Any}() end
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

function get_agent(state, agent)
    sym, uid = agent
    return state.pop[sym][uid]
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

function update_dtime!(time, srx, deleted, agents)
    for agent in deleted
        setfield!(agent, :dtime, time)
        setfield!(agent, :dinteraction, srx.uid)
    end
end

function log_outstates!(srx::SimulationReaction, state, rxtime, agents, model, results::SimulationResults)
    saving = srx.pitx.itxdef.saving
    out_traits = filter(x -> x isa SaveOutStateTrait, saving) 
    isempty(out_traits) && return nothing 

    savevalues = Dict()

    for agent in Iterators.flatten(values.(values(agents)))
        for save in out_traits 
            !isequal(save.agent, agent.sym) && continue
            name = Symbol(string(save_trait_name(save)) * "_$(srx.pitx.itxdef.name)")

            idx = indexof(save.trait, unknowns(model.traitdefs[agent.sym].dynamics))

            !in(name, keys(savevalues)) && begin 
                savevalues[name] = Float64[]
            end
            push!(savevalues[name], Float64(agent.init_trait[idx][2]))
        end
    end

    for name in keys(savevalues)
        !in(name, keys(results.outstates)) && begin 
            results.outstates[name] = DiffEqArray([savevalues[name], ], [rxtime, ]) 
            continue 
        end
        push!(results.outstates[name].t, rxtime)
        push!(results.outstates[name].u, savevalues[name])
    end
end

function log_instates!(srx::SimulationReaction, state, rxtime, agents, model, results::SimulationResults)
    saving = srx.pitx.itxdef.saving
    in_traits = filter(x -> x isa SaveInStateTrait, saving) 
    isempty(in_traits) && return nothing 

    savevalues = Dict()

    for agent in agents
        for save in in_traits 
            !isequal(save.agent, agent.sym) && continue
            name = Symbol(string(save_trait_name(save)) * "_$(srx.pitx.itxdef.name)")

            idx = indexof(save.trait, unknowns(model.traitdefs[agent.sym].dynamics))

            !in(name, keys(savevalues)) && begin 
                savevalues[name] = Float64[]
            end
            push!(savevalues[name], agent.simulation(rxtime)[idx])
        end
    end

    for name in keys(savevalues)
        !in(name, keys(results.instates)) && begin 
            results.instates[name] = DiffEqArray([savevalues[name], ], [rxtime, ]) 
            continue 
        end
        push!(results.instates[name].t, rxtime)
        push!(results.instates[name].u, savevalues[name])
    end
end

function log_snapshot!(time, saving, state::SimulationState, model, results::SimulationResults)
    isempty(saving) && return nothing

    for save in saving 
        snapshot = Float64[]
        snapshot_n = [] 
        name = save_trait_name(save) 

        for agent in Iterators.flatten(values.(values(state.pop)))
            if save isa StateSnapshot 
                isa(model.traitdefs[agent.sym].dynamics, EmptyTraitProblem) && continue
                !isequal(agent.sym, save.agent) && continue
                push!(snapshot, agent.simulation(time; idxs=save.trait)[1])
            elseif save isa PopulationSnapshot
                isequal(agent.sym, save.agent) ? push!(snapshot_n, agent.sym) : nothing
            end 
        end
        
        if save isa StateSnapshot
            !haskey(results.snapshot, name) && begin 
                results.snapshot[name] = DiffEqArray([snapshot, ], [time, ]) 
                continue
            end
            push!(results.snapshot[name].t, time)
            push!(results.snapshot[name].u, snapshot)
        elseif save isa PopulationSnapshot
            !haskey(results.snapshot, name) && begin 
                results.snapshot[name] = DiffEqArray(Float64[length(snapshot_n), ], Float64[time, ]) 
                continue
            end
            push!(results.snapshot[name].t, time)
            push!(results.snapshot[name].u, length(snapshot_n))
        end
    end
end

function initialise_agents(model, init_pop, tspan, params::SimulationParameters; kwargs...) 
    # Make the population state dictionary.
    pop = Dict{Num, IdDict{idType, Any}}(
        Num(s) => IdDict{idType, Any}() for s in unknowns(model.rn))
    for (agent, init_traits) in init_pop 
        dyn = model.traitdefs[agent].dynamics
        cts = model.traitdefs[agent].constants

        c = Tuple(x => Symbolics.unwrap.(substitute([x, ], init_traits)...) for x in cts)
        tr = Tuple(x => Symbolics.unwrap.(substitute([Num(x), ], init_traits)...) for x in unknowns(dyn))

        agent_ = AgentState(tspan[1], agent, tr, c, Vector{Tuple{Num, idType}}(), nothing)
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

function save_interactions!(interactions, rxtime, srx, state)
    agents = [(agent[2].sym, agent[1]) for agent in Iterators.flatten(values(state.pop))]
    push!(interactions, (rxtime, srx, agents))
end

function simulate(modeldef::AgentsModel, init_pop, params::SimulationParameters; 
    showprogress=true, 
    save_interactions=false,
    trace_agents=false) 

    state, results, model = init_simulator(modeldef, init_pop, params)
    
    showprogress && begin
        progress = ProgressUnknown()
    end

    all_agents = Dict{Num, Dict{idType, Any}}()
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
                update_dtime!(next_rx_time, srx, deleted_agents, state.pop)

                # Logging
                log_instates!(srx, state, next_rx_time, deleted_agents, model, results)

                # Remove agents involved in the current reaction and reactions with
                # them as substrates.
                filter_rxs!(state, deleted_agents)

                # Simulate traits of the new agents to tend.   
                simulate_traits!(new_agents, next_rx_time, tend, params; model=model)

                log_outstates!(srx, state, next_rx_time, new_agents, model, results)

                # Add the new to the population state.
                trace_agents && push_to_pop!(all_agents, deleted_agents)
                push_to_pop!(state.pop, new_agents)

                # New reactions.
                make_reactions!(new_agents, state, model, (next_rx_time, tend), params; make_zero_substrate_rx=false)
                state.t = next_rx_time 
                save_interactions && save_interactions!(results.interactions, next_rx_time, srx, state)
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
      
            trace_agents && push_to_pop!(all_agents, state.pop)
            results.agents = all_agents 
            results.final_pop = state.pop
            return results
        else
            rethrow(e)
        end
    end

    log_snapshot!(state.t, params.snapshot, state, model, results)
    results.tend = state.t

    trace_agents && push_to_pop!(all_agents, state.pop)
    results.agents = all_agents 
    results.final_pop = state.pop
    return results
end
