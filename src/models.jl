abstract type AbstractParameterCnx end

#SpecifiedNumerics = Union{Real, Matrix{Real}, Array{Real}}

struct ParameterCnx{pType, iType} <: AbstractParameterCnx
    parameter::pType
    idx::iType
end

struct Variable{pType,fType} <: AbstractParameterCnx
    parameter::pType
    symbf::fType
end

function variable_subs(vars, pstate, symbs)
    pstatesubs = Any[x => y for (x, y) in zip(symbs, pstate)]
    varstosub = Dict(var.parameter => var.symbf for var in vars)

    out = [] 
    n = 1

    while true
        isempty(keys(varstosub)) && break
        n > length(vars) && begin
            error("Substitution of variables failed to resolve symbols $(collect(keys(varstosub))). This could result from unspecified parameter values or partially specified transitions.")
        end

        for key in keys(varstosub) 
            # Substitute variable values in pstatesubs to variable expression. 
            # If no symbolic variables add the var to pstatesubs and output.
            # Keep iterating till no symbolic variables left.
            val = varstosub[key](pstatesubs)
            vs = Symbolics.get_variables(val)

            isempty(vs) && begin
                push!(pstatesubs, key => val)
                push!(out, key => val)
                pop!(varstosub, key)
            end
        end
        n += 1
    end
    return out 
end

function replace_with_connection(exprs, cnx)
    replacements = Dict(c.trait => c.parameter for c in cnx)
    return [substitute(e, replacements) for e in Num.(exprs)]
end

struct TraitTransition{R}
    rule::R
end

struct TransitionDef{rxType,mType,ttType}
    rx::rxType
    method::mType
    traitt::ttType
end
Base.hash(tdef::TransitionDef) = hash(tdef.rx)

struct EmptyTraitProblem 
    function EmptyTraitProblem(;kwargs...)
        return new()
    end
end
Catalyst.unknowns(::EmptyTraitProblem) = ()
TraitProblems = Union{JumpProblem, ODEProblem, EmptyTraitProblem}
CommonSolve.solve(prob::EmptyTraitProblem, args...; kwargs...) = nothing

struct PopulationItxDef{nType,rxType,sType,pType,cType,vType,svType} 
    name::nType
    rx::rxType
    species::sType
    params::pType
    cnx::cType
    vars::vType
    saving::svType

    function PopulationItxDef(rx::rxType, species::sType, ps::pType, cnx::cType, vars::vType; saving::svType=[], name::nType) where {rxType,sType,pType,cType,vType,svType,nType}
        return new{nType,rxType,sType,pType,cType,vType,svType}(name, rx, species, ps, cnx, vars, saving)
    end
end
Base.hash(pitxd::PopulationItxDef) = hash(pitxd.rx)

struct PopulationItx{iType, F1, F2, F3, psType, pvType, pmType, srType}
    itxdef::iType
    ratef::F1
    ratefmax::F2
    Lf::F3
    ispopdep::Bool
    uid::UInt
    psymbs::psType
    pvec::pvType
    pmod::pmType
    subsrules::srType
end

function PopulationItx(itxdef::PopulationItxDef{nType,rxType,sType,pType,cType,vType,svType}, model, params) where {nType,rxType,sType,pType,cType,vType,svType}
    modelrn = deepcopy(model.rn)
    ratef = _gen_rate_function(deepcopy(itxdef.rx.rx.rate), modelrn)
    ratefmax = _gen_rate_function(get_λmax(deepcopy(itxdef.rx.method)), modelrn)
    Lf = _gen_rate_function(get_L(deepcopy(itxdef.rx.method)), modelrn)

    ispopdep = false

    if !isempty(union(ModelingToolkit.get_variables(itxdef.rx.rx.rate), unknowns(modelrn)))
        ispopdep = true
    end

    subsrules_ = Dict{Num, Tuple{Int, Num, Tuple{Bool, Int64}}}()
    for cx in itxdef.cnx
        subs = reduce(vcat, fill.(itxdef.rx.rx.substrates, itxdef.rx.rx.substoich))
        type = subs[cx.idx]
        for (param, trait) in cx.parameter
            subsrules_[param] = (cx.idx, trait, model.traits[type].symtoidx[trait])
        end
    end

    psymb = union(parameters(modelrn), itxdef.params...)

    ps = replace(Num.(psymb), params.ps...)
    ps = Symbolics.unwrap.(ps)

    pvec = Vector{Float64}(undef, length(ps))
    # Symbol of param, index in the params vector, (index of substrate, trait sym, (isdiscrete, index in simulation))
    pmod = Tuple{Num, Int, Tuple{Int, Num, Tuple{Bool, Int64}}}[]

    for (i,p) in enumerate(ps)
        if (p isa Real) 
            pvec[i] = p
            continue
        elseif haskey(subsrules_, p)
            push!(pmod, (Num(p), i, subsrules_[Num(p)]))
        else
            pvec[i] = 0.0
        end
    end

    pitxdeftype = PopulationItxDef{nType,rxType,sType,pType,cType,vType,svType}
    return PopulationItx{
        pitxdeftype,typeof(ratef),typeof(ratefmax),typeof(Lf),typeof(psymb),typeof(pvec),typeof(pmod),typeof(subsrules_)}(
            itxdef, 
            ratef, 
            ratefmax, 
            Lf,
            ispopdep, 
            hash(itxdef), 
            psymb,
            pvec,
            pmod,
            subsrules_)
end

Base.isequal(pitxa::PopulationItx, pitxb::PopulationItx) = isequal(pitxa.uid, pitxb.uid)
Base.hash(pitx::PopulationItx) = pitx.uid
getuid(pitx::PopulationItx)::UInt = pitx.uid

function process_interaction(inter::PopulationItxDef, model, params) 
    return PopulationItx(inter, model, params)
end

function Base.show(io::IO, itx::PopulationItx{M,F}) where {M,F} 
    print(io, "PopulationItx $(itx.itxdef.rx.rx).")
end

function trait_transition(pitx::itxType, products::pType, substrates::sType, subsrules::srType, state::stType, model::mType, t::Float64) where {itxType, pType, sType, srType, stType, mType}
    subs_ = Pair{Num, Float64}[]
    out_ = Vector{Pair{Num, Num}}[]
    for (s, (idx_, sym_, la_)) in subsrules
        push!(subs_, s => get_trait_value(substrates[idx_], t, la_))
    end
    
    isnothing(pitx.itxdef.rx.traitt.rule) && return tuple()

    subs_dict = Dict{Num, Float64}(subs_)

    for rr in pitx.itxdef.rx.traitt.rule
        push!(out_, [first.(rr)...] .=> Symbolics.substitute.(last.(rr), Ref(subs_dict)))
    end
    return out_
end

function pstate!(pmod, pvec, subsrules, model, substrates, state, t::Float64)
    isempty(pmod) && return nothing 
    for (p, i, (idx_, sym_, la_)) in pmod
        @inbounds pvec[i] = get_trait_value(substrates[idx_], t, la_)
    end
end

function pstate(pmod, subsrules, model, substrates, state, t::Float64)
    pvec = zeros(Float64, length(pmod)) 
    isempty(pmod) && return nothing 
    for (p, i, (idx_, sym_, la_)) in pmod
        @inbounds pvec[i] = get_trait_value(substrates[idx_], t, la_)
    end
    return pvec
end

struct HybridSDEDynamics{cType, dType}
    continuous::cType
    discrete::dType
end

function ModelingToolkit.unknowns(hybrid::HybridSDEDynamics)
    uks = unknowns(hybrid.continuous)
#    filter(x -> !ModelingToolkit.isbrownian(x), uks)
end

function ModelingToolkit.parameters(hybrid::HybridSDEDynamics)
#    return unique([parameters(hybrid.continuous)..., parameters(hybrid.discrete)...])
end

struct AgentDynamics{D,N}
    dynamics::D
    constants::NTuple{N, Num}
    symtoidx::Dict{Num, Tuple{Bool, Int}} 
end

function Catalyst.extend(cont::SDESystem, disc::ReactionSystem)
    @error "Extending SDE with reaction network currently requires the following workaround: specify 
        HybridSDEDynamics(continuous::SDESystem, discrete::ReactionSystem) as the agent dynamics and construct
        the AgentDynamics struct by calling AgentDynamics((hybrid_sde, ), constants)."
end

function AgentDynamics(dynamics, constants) 
    keys = Num[]
    vals = Tuple{Bool, Int}[]
    for (i, c) in enumerate(constants)
        push!(keys, c)
        push!(vals, (true, i))
    end

    for (i, c) in enumerate(unknowns(dynamics))
        push!(keys, c)
        push!(vals, (false, i))
    end
    AgentDynamics{typeof(dynamics),length(constants)}(dynamics, constants, Dict(keys .=> vals))
end

struct Trait{T}
    symb::Num
    problem::T
    symtoidx::Dict{Num, Tuple{Bool, Int}} # tuple element true if constant
end

struct AgentsModel{rnType, rxType, trType}
    rn::rnType
    rxs::rxType
    traits::trType
    function AgentsModel(rxs::rxType, traits::trType) where {rxType,trType}
        rxs_ = Union{Equation, Reaction}[] 
        bnd_ = Union{Equation, Reaction}[] 
        sps_ = []
        params_ = []
        for r in rxs
            push!(rxs_, r.rx.rx)
            push!(sps_, r.species...)
            Catalyst.get_variables!(params_, get_λmax(r.rx.method))
            Catalyst.get_variables!(params_, get_L(r.rx.method))
        end

        @named rn_ = ReactionSystem(rxs_)
        @named rn = ReactionSystem(
            rxs_, 
            Catalyst.get_iv(rn_), 
            setdiff(union(Catalyst.get_species(rn_), sps_, collect(keys(traits))), [Catalyst.get_iv(rn_),]), 
            setdiff(union(Catalyst.parameters(rn_), params_), [Catalyst.get_iv(rn_), Catalyst.get_species(rn_)...]))
        return new{typeof(rn),rxType,trType}(rn, rxs, traits)
    end
end

struct PopulationModel{rnType,rxType,tpType,tdType}
    rn::rnType
    rxs::rxType
    traitprobs::tpType
    traitdefs::tdType

    function PopulationModel(popmodeldef::amType, params) where {amType}
        trait_problems = make_trait_problems(popmodeldef, params)
        itxs = [process_interaction(rx, popmodeldef, params) for rx in popmodeldef.rxs]
        return new{typeof(popmodeldef.rn),typeof(itxs),typeof(trait_problems),typeof(popmodeldef.traits)}(
            popmodeldef.rn, itxs, trait_problems, popmodeldef.traits) 
    end
end

struct Indexing{N}
    index::Int64
    parent::NTuple{N, Int64}

    function Indexing(index, parent)
        new{length(parent)}(index, parent)
    end
end

function make_trait_problems(model::AgentsModel, params;)
    Dict{Num, Trait}(
        trait.first => make_trait_problem(trait.first, trait.second, params.tspan, params.ps; 
            jumpaggregator=params.jumpaggregator, params.solverkws...) for trait in model.traits)
end

Problems = Union{ODEProblem, SDEProblem, JumpProblem}
Systems = Union{ODESystem, SDESystem, JumpSystem}
ProblemSystemDict = Dict(ODESystem => ODEProblem, SDESystem => SDEProblem, JumpSystem => JumpProblem)

function make_trait_problem(sym, dynamics::AgentDynamics{S, N}, tspan, ps; kwargs...) where {S <: Systems, N}
    keys = Num[]
    vals = Tuple{Bool, Int}[]
    for (i, c) in enumerate(dynamics.constants)
        push!(keys, c)
        push!(vals, (true, i))
    end
   
    for (i, c) in enumerate(unknowns(dynamics.dynamics))
        push!(keys, c)
        push!(vals, (false, i))
    end

    Trait(sym, ProblemSystemDict[S]{true}(complete(dynamics.dynamics), zeros(length(unknowns(dynamics.dynamics))), tspan, ps), Dict(keys .=> vals))
end

function make_trait_problem(sym, dynamics::AgentDynamics{HybridSDEDynamics{cType, dType}, N}, tspan, ps; jumpaggregator) where {cType, dType, N}
    keys = Num[]
    vals = Tuple{Bool, Int}[]
    for (i, c) in enumerate(dynamics.constants)
        push!(keys, c)
        push!(vals, (true, i))
    end
   
    for (i, c) in enumerate(unknowns(dynamics.dynamics))
        push!(keys, c)
        push!(vals, (false, i))
    end

    prob = make_hybrid(dynamics.dynamics, zeros(length(unknowns(dynamics.dynamics))), tspan, ps; jumpaggregator=jumpaggregator)
    Trait(sym, prob, Dict(keys .=> vals))
end


function make_trait_problem(sym, dynamics::AgentDynamics{ReactionSystem{T}, N}, tspan, ps; jumpaggregator) where {T,N}
    keys = Num[]
    vals = Tuple{Bool, Int}[]
    for (i, c) in enumerate(dynamics.constants)
        push!(keys, c)
        push!(vals, (true, i))
    end
   
    for (i, c) in enumerate(unknowns(dynamics.dynamics))
        push!(keys, c)
        push!(vals, (false, i))
    end

    isempty(setdiff(equations(dynamics.dynamics), reactions(dynamics.dynamics))) && begin 
        dyn = dynamics.dynamics
        jin = JumpInputs(dyn, zeros(length(unknowns(dyn))), tspan, ps)
        return Trait(
            sym, 
            JumpProblem(jin), Dict(keys .=> vals))
    end

    prob = make_hybrid(dynamics.dynamics, zeros(length(unknowns(dynamics.dynamics))), tspan, ps; jumpaggregator=jumpaggregator)
    Trait(sym, prob, Dict(keys .=> vals))
end

function make_trait_problem(sym, dynamics::AgentDynamics{EmptyTraitProblem, N}, tspan, ps; kwargs...) where {N}
    keys = Num[]
    vals = Tuple{Bool, Int}[] 
    for (i, c) in enumerate(dynamics.constants)
        push!(keys, c)
        push!(vals, (true, i))
    end
    Trait(sym, dynamics.dynamics, Dict(keys .=> vals))
end

function make_hybrid(rs, init, tspan, params; 
        jumpaggregator, 
        name = nameof(rs),
        checks = false,
        combinatoric_ratelaws=Catalyst.get_combinatoric_ratelaws(rs),
        include_zero_odes=true)
    # Temporary fix workaround. Remove when hybrid systems supported by Catalyst.
    
    flatrs = Catalyst.flatten(rs)
    eqs = Any[assemble_hybrid_jumps(flatrs)...]
    ists, ispcs = Catalyst.get_indep_sts(flatrs)
    eqs, us, ps, obs, defs = Catalyst.addconstraints!(eqs, flatrs, ists, ispcs; 
        remove_conserved = false)

    jsys = JumpSystem(eqs, get_iv(flatrs), us, ps;
            observed = obs,
            name,
            defaults = MT._merge(Dict(), MT.defaults(flatrs)),
            checks,
            discrete_events = MT.discrete_events(flatrs),
            continuous_events = MT.continuous_events(flatrs),)

    u0map = symmap_to_varmap(rs, init)
    pmap = symmap_to_varmap(rs, params)

    prob = ODEProblem(complete(jsys), u0map, tspan, pmap; )
    jprob = JumpInputs(complete(jsys), prob)
    return JumpProblem(jprob)
end


function make_hybrid(hdyn::HybridSDEDynamics, init, tspan, params; 
        jumpaggregator, 
        name = nameof(hdyn.discrete),
        checks = false,
        combinatoric_ratelaws=Catalyst.get_combinatoric_ratelaws(hdyn.discrete),
        include_zero_odes=true)
    
    # Temporary fix workaround. Remove when hybrid systems supported by Catalyst.
    eqs = equations(hdyn.continuous)
    rxs = reactions(hdyn.discrete)  

    @named rs = ReactionSystem(
        [rxs; eqs], 
        ModelingToolkit.get_iv(hdyn.discrete), 
        filter(x -> !ModelingToolkit.isbrownian(x), unknowns(hdyn)),
        first.(params))

    @named sde = SDESystem(
        eqs,
        vcat(hdyn.continuous.noiseeqs...),
        ModelingToolkit.get_iv(hdyn.continuous),
        filter(x -> !ModelingToolkit.isbrownian(x), unknowns(rs)),
        first.(params))

    flatrs = Catalyst.flatten(rs)
    eqs = Any[assemble_hybrid_jumps(flatrs)...]
    ists, ispcs = Catalyst.get_indep_sts(flatrs)
    _, us, ps, obs, defs = Catalyst.addconstraints!(eqs, flatrs, ists, ispcs; 
        remove_conserved = false)

    jsys = JumpSystem(eqs, get_iv(flatrs), us, ps;
        observed = obs,
        name,
        defaults = MT._merge(Dict(), MT.defaults(flatrs)),
        checks,
        discrete_events = MT.discrete_events(flatrs),
        continuous_events = MT.continuous_events(flatrs),)

    u0map = symmap_to_varmap(rs, init)
    pmap = symmap_to_varmap(rs, params)

    prob = SDEProblem(complete(sde), u0map, tspan, pmap;)
    JumpProblem(complete(jsys), prob, jumpaggregator)
end

let x = Threads.Atomic{Int}(0)
    mutable struct AgentState{tType, biType, sType, pType, inType, cType}
        sym::sType
        btime::tType
        dtime::tType
        binteraction::biType
        dinteraction::Union{UInt, Nothing}
        parents::pType
        srxs::Vector{Any}
        uid::UInt
        init_trait::inType
        consts::cType
        simulation::Union{Nothing, ODESolution, RODESolution}

        function AgentState(btime::tType, sym::sType, init_trait::inType, consts::cType, parents::pType, binteraction::biType) where {tType, biType, sType, inType, cType, pType}
            atomic_add!(x,1)

            new{tType, biType, sType, pType, inType, cType}(
                sym,
                btime,
                typemax(btime),
                binteraction,
                nothing,
                parents,
                Vector{Tuple{UInt, UInt}}(),
                hash(sym, hash(x.value)),
                init_trait,
                consts,
                nothing)
        end
    end
end

Base.isequal(a::AgentState, b::AgentState) = isequal(a.uid, b.uid)
getsim(agent::AgentState, t::Float64) = agent.simulation(t)
getsymid(agent::AgentState) = (agent.sym, agent.uid) 

function get_trait_value(agent::AgentState, t::Float64, pair)::Float64
    # pair = (Bool, Int) where pair[1] is whether the trait is constant and
    # pair[2] is the index of the trait in a simulation.
    pair[1] && return last(agent.consts[pair[2]])
    return @inbounds agent.simulation(t; continuity = :right)[pair[2]]
end

function Base.show(io::IO, agent::AgentState)
    print(io, "Agent of type $(agent.sym)")
end

get_sym(agent::AgentState) = agent.sym
get_sym_string(agent::AgentState) = string(agent.sym.f)
get_parents(agent::AgentState) = agent.parents
get_birth(agent::AgentState) = agent.btime

function update_dtime!(time, agent::AgentState)
    agent.dtime = time
end

struct SimulationReaction{pType,mType,sType,uType}
    pitx::pType
    substrates::sType
    sampler::mType
    uid::uType

    function SimulationReaction(pitx::pType, substrates, sampler::mType) where {pType,mType}
        subs = getsymid.(substrates)
        uid = hash(substrates, pitx.uid)

        for r in substrates
            push!(r.srxs, (getuid(pitx), uid))
        end

        return new{pType,mType,typeof(subs),typeof(uid)}(pitx, subs, sampler, uid)
    end
end

Base.isequal(srxa::SimulationReaction, srxb::SimulationReaction) = isequal(srxa.uid, srxb.uid)
Base.hash(srxa::SimulationReaction) = srxa.uid

function Base.show(io::IO, srx::SimulationReaction)
    print(io, "Simulation reaction with interaction $(srx.pitx)")
end

function compute_input(itx::PopulationItx, agents, model::PopulationModel, time)
    return itx.trait_inp_f(agents, time)
end

function compute_new_traits(itx::PopulationItx, input)
    return itx.trait_f(input) 
end
