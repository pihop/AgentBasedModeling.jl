abstract type AbstractParameterCnx end

struct ParameterCnx{pType, iType} <: AbstractParameterCnx
    parameter::pType
    idx::iType
end

struct Variable{pType,fType} <: AbstractParameterCnx
    parameter::pType
    symbf::fType
end

function variable_subs(vars, pstate, symbs)
    pstatesubs = [x => y for (x, y) in zip(symbs, pstate)]
    return [var.parameter => var.symbf(pstatesubs) for var in vars]
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

function AgentDynamics(dynamics::Union{Vector,Tuple}, constants) 
    if length(dynamics) > 1
        dynamics_ = extend(dynamics...)
    elseif length(dynamics) == 1
        dynamics_ = dynamics[1]
    end

    keys = Num[]
    vals = Tuple{Bool, Int}[]
    for (i, c) in enumerate(constants)
        push!(keys, c)             
        push!(vals, (true, i))             
    end

    for (i, c) in enumerate(unknowns(dynamics_))
        push!(keys, c)             
        push!(vals, (false, i))             
    end
    AgentDynamics{typeof(dynamics_),length(constants)}(dynamics_, constants, Dict(keys .=> vals))
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

function make_trait_problem(sym, dynamics::AgentDynamics{HybridSDEDynamics, N}, tspan, ps; jumpaggregator) where {N}
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
        dprob = DiscreteProblem(
            dynamics.dynamics, zeros(length(unknowns(dynamics.dynamics))), tspan, ps)
        return Trait(
            sym, 
            JumpProblem(dynamics.dynamics, dprob, jumpaggregator), Dict(keys .=> vals))
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

function make_hybrid(trait, init, tspan, ps; jumpaggregator)
    eqs = setdiff(equations(trait), reactions(trait))

    @named rn = ReactionSystem(
        reactions(trait), 
        ModelingToolkit.get_iv(trait), 
        filter(x -> !ModelingToolkit.isbrownian(x), unknowns(trait)), 
        parameters(trait))

    @named odes = ODESystem([eqs...], 
        ModelingToolkit.get_iv(trait), unknowns(trait), parameters(trait);)
    jsys = convert(JumpSystem, complete(rn))

    oprob = ODEProblem(complete(odes), init, tspan, ps;)
    JumpProblem(complete(jsys), oprob, jumpaggregator; save_positions = (true,true))
end

function make_hybrid(trait::HybridSDEDynamics, init, tspan, ps; jumpaggregator)
    eqs = equations(trait.continuous)
    rxs = reactions(trait.discrete)  

    @named rn = ReactionSystem(
        rxs, 
        ModelingToolkit.get_iv(trait.discrete), 
        filter(x -> !ModelingToolkit.isbrownian(x), unknowns(trait)),
        first.(ps))

    jsys = convert(JumpSystem,  complete(rn))
   
    @named sde = SDESystem(
        eqs,
        vcat(trait.continuous.noiseeqs...),
        ModelingToolkit.get_iv(trait.continuous),
        filter(x -> !ModelingToolkit.isbrownian(x), unknowns(trait)),
        first.(ps))

    jsys = convert(JumpSystem, complete(rn))

    oprob = SDEProblem(complete(sde), init, tspan, ps;) 
    JumpProblem(complete(jsys), oprob, jumpaggregator; )
end

let x = Threads.Atomic{Int}(0)
    mutable struct AgentState{tType, sType, pType, inType, cType}
        sym::sType
        btime::tType
        dtime::Union{tType, Nothing}
        idx::Int64
        parents::pType
        srxs::Vector{Any}
        uid::UInt
        init_trait::inType
        consts::cType
        simulation::Union{Nothing, ODESolution, RODESolution}
#        simulation_interp
        trait_snapshot::Union{Vector{Float64}, Nothing}

        function AgentState(btime::tType, sym::sType, init_trait::inType, consts::cType, parents::pType) where {tType, sType, inType, cType, pType}
            atomic_add!(x,1)

            new{tType, sType, pType, inType, cType}(
                sym,
                btime,
                nothing,
                x.value,
                parents,
                Vector{Tuple{UInt, UInt}}(),
#                Vector{Any}(),
                hash(sym, hash(x.value)),
                init_trait,
                consts,
                nothing,
                nothing)
        end
    end
end

Base.isequal(a::AgentState, b::AgentState) = isequal(a.uid, b.uid)
getsim(agent::AgentState, t::Float64) = agent.simulation(t)
@inline getsymid(agent::AgentState)::Tuple{Num, UInt} = (agent.sym, agent.uid) 

function update_trait_snapshot!(agent::AgentState, t::Float64)
    isnothing(agent.simulation) && return nothing
    agent.trait_snapshot = agent.simulation(t)
end

function get_trait_value(agent::AgentState, t::Float64, pair)::Float64
    pair[1] && return last(agent.consts[pair[2]])
    return @inbounds agent.trait_snapshot[pair[2]]
end

function Base.show(io::IO, agent::AgentState)
    print(io, "Agent of type $(agent.sym)")
end

get_sym(agent::AgentState) = agent.sym
get_sym_string(agent::AgentState) = string(agent.sym.f)
get_id(agent::AgentState) = agent.idx
get_parents(agent::AgentState) = agent.parents
get_birth(agent::AgentState) = agent.btime

function update_dtime!(time, agent::AgentState)
    agent.dtime = time
end

struct SimulationReaction{pType,mType,sType,uType}
    pitx::pType
#    substrates::NTuple{N3, AgentState}
    substrates::sType
    sampler::mType
    uid::uType
end

#function SimulationReaction(pitx::pType, substrates, sampler::mType) where {pType,mType}
function SimulationReaction(pitx::pType, substrates, sampler::mType) where {pType,mType}
    subs = getsymid.(substrates)
    uid = hash(substrates, pitx.uid)

    for r in substrates
        push!(r.srxs, (pitx.uid, uid))
    end

    return SimulationReaction{pType,mType,typeof(subs),typeof(uid)}(pitx, subs, sampler, uid)
#    return srx
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
