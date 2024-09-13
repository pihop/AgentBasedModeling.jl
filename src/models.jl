abstract type AbstractParameterCnx end

struct ParameterCnx <: AbstractParameterCnx
    parameter::Num
    type::Num 
    trait::Union{Num, Symbol}
end

struct Variable{T,F} <: AbstractParameterCnx
    parameter::T
    symbf::F
end

function variable_subs(vars, pstate, symbs)
    pstatesubs = [x => y for (x, y) in zip(symbs, pstate)]
    return [var.parameter => var.symbf(pstatesubs) for var in vars]
end

function replace_with_connection(exprs, cnx)
    replacements = Dict(c.trait => c.parameter for c in cnx)
    return [substitute(e, replacements) for e in Num.(exprs)]
end

struct AgeConnection <: AbstractParameterCnx
    parameter::Num
    type::Num
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

struct PopulationItxDef{nType,mType,N,M,cType,sType} 
    name::nType
    rx::TransitionDef
    species::NTuple{N,Num}
    params::NTuple{M,Num}
    cnx::cType
    vars::Vector{Variable}
    saving::sType

    function PopulationItxDef(rx, species, ps, cnx, vars; saving=[], name) #where {C<:AbstractParameterCnx}
        return new{typeof(name), typeof(rx.method),length(species), length(ps), typeof(cnx),typeof(saving)}(
            name, rx, tuple(species...), tuple(ps...), cnx, vars, saving)
    end
end
Base.hash(pitxd::PopulationItxDef) = hash(pitxd.rx)

struct PopulationItx{mType,N,cType,sType,F1,F2,F3,N2} 
    itxdef::PopulationItxDef
    ratef::F1
    ratefmax::F2
    Lf::F3
    ispopdep::Bool
    uid::UInt
    psymbs::Vector{Num}
    pvec::Vector{Float64}
    pmod::NTuple{N2,Tuple{Num, Int, Tuple{Int, Num, Num, Tuple{Bool, Int64}}}}
    subsrules::Dict{Num, Tuple{Int, Num, Num, Tuple{Bool, Int64}}}
end

function PopulationItx(itxdef::PopulationItxDef{nType,mType,N,cType,sType}, model, params) where {nType,mType,N,cType,sType}
    modelrn = deepcopy(model.rn)
    ratef = _gen_rate_function(deepcopy(itxdef.rx.rx.rate), modelrn)
    ratefmax = _gen_rate_function(get_λmax(deepcopy(itxdef.rx.method)), modelrn)
    
    Lf = _gen_rate_function(get_L(deepcopy(itxdef.rx.method)), modelrn)

    ispopdep = false
    
    if !isempty(union(ModelingToolkit.get_variables(itxdef.rx.rx.rate), unknowns(modelrn)))
        ispopdep = true
    end

    subsrules_ = Dict{Num, Tuple{Int, Num, Num, Tuple{Bool, Int64}}}()
    for cx in itxdef.cnx
        idx_ = findall(x -> isequal(x, cx.type), itxdef.rx.rx.substrates)
        length(idx_) > 1 && throw("Substitution not uniquely defined") 
        subsrules_[cx.parameter] = (idx_[1], cx.type, cx.trait, model.traits[cx.type].symtoidx[cx.trait])
    end

    psymb = union(parameters(modelrn), itxdef.params...)

    ps = replace(Num.(psymb), params.ps...)
    ps = Symbolics.unwrap.(ps)

    pvec = Vector{Float64}(undef, length(ps))
    pmod = Tuple{Num, Int, Tuple{Int, Num, Num, Tuple{Bool, Int64}}}[]

    for (i,p) in enumerate(ps)
        if (p isa Float64) 
            pvec[i] = p
            continue
        elseif haskey(subsrules_, p)
            push!(pmod, (Num(p), i, subsrules_[Num(p)]))
        else
            pvec[i] = 0.0
        end
    end
    
    return PopulationItx{
        mType,N,cType,sType,typeof(ratef),typeof(ratefmax),typeof(Lf),length(pmod)}(
            itxdef, 
            ratef, 
            ratefmax, 
            Lf,
            ispopdep, 
            hash(itxdef), 
            psymb,
            pvec,
            tuple(pmod...),
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

function trait_transition(pitx, products, substrates, subsrules, state, model, t::Float64)
    subs_ = Pair{Num, Float64}[]
    out_ = []
    for (s, (idx_, type_, sym_, la_)) in subsrules
        push!(subs_, s => get_trait_value(substrates[idx_], t, la_))
    end
#    display(subs_)
    
    isnothing(pitx.itxdef.rx.traitt.rule) && return []

    for rr in pitx.itxdef.rx.traitt.rule
        push!(out_, first.(rr) .=> Symbolics.substitute.(last.(rr), Ref(Dict(subs_...))))
    end
    return out_
end

function pstate!(pmod, pvec, subsrules, model, substrates, state, t::Float64)
    isempty(pmod) && return nothing 
    for (p, i, (idx_, type_, sym_, la_)) in pmod
        @inbounds pvec[i] = get_trait_value(substrates[idx_], t, la_)
    end
end

struct HybridSDEDynamics
    continuous
    discrete
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

function AgentDynamics(dynamics::Union{Vector,Tuple}, constants) 
    if length(dynamics) > 1
        try  
            dynamics_ = extend(dynamics...)
        catch
            @error "Extending SDE with reaction network currently requires the following workaround: specify 
                HybridSDEDynamics(continuous::SDESystem, discrete::ReactionNetwork) as the agent dynamics and construct
                the AgentDynamics struct by calling AgentDynamics((hybrid_sde, ), constants)."
        end
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

struct PopulationModelDef
    rn::ReactionSystem
    rxs::Vector{PopulationItxDef}
    traits::Dict{Num, AgentDynamics} 
    function PopulationModelDef(rxs, traits)
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
            setdiff(union(Catalyst.get_species(rn_), sps_), [Catalyst.get_iv(rn_),]), 
            setdiff(union(Catalyst.parameters(rn_), params_), [Catalyst.get_iv(rn_), Catalyst.get_species(rn_)...]))
        return new(rn, rxs, traits)
    end
end

struct PopulationModel
    rn::ReactionSystem
    rxs::Vector{PopulationItx}
    traitprobs::Dict{Num, Trait}
    traitdefs::Dict{Num, AgentDynamics}

    function PopulationModel(popmodeldef::PopulationModelDef, params)
        trait_problems = make_trait_problems(popmodeldef, params)
        itxs = PopulationItx[process_interaction(rx, popmodeldef, params) for rx in popmodeldef.rxs]
        return new(popmodeldef.rn, itxs, trait_problems, popmodeldef.traits) 
    end
end

struct Indexing{N}
    index::Int64
    parent::NTuple{N, Int64}
    
    function Indexing(index, parent)
        new{length(parent)}(index, parent)        
    end
end

function make_trait_problems(model::PopulationModelDef, params;)
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

function make_trait_problem(sym, dynamics::AgentDynamics{HybridSDEDynamics, N}, tspan, ps; jumpaggregator) where {T,N}
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

    @named odes = ODEProblem([eqs...], 
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
    mutable struct AgentState{P,N1,N2,idType}
        sym::Num
        btime::Float64
        dtime::Union{Float64, Nothing}
        idx::Int64
        parents::P
        srxs::Vector{Tuple{PopulationItx, UInt}}
        uid::idType#UInt
        init_trait::NTuple{N1, Pair{Num,Float64}}
        consts::NTuple{N2, Pair{Num,Float64}}
        simulation::Union{Nothing, ODESolution, RODESolution}
        simulation_interp

        function AgentState(btime, sym, init_trait, consts, parents::P) where {P}
            atomic_add!(x,1)
#            uid = uuid4()
            agent = new{P,length(init_trait),length(consts),idType}()
            agent.btime = btime
            agent.sym = sym
            agent.idx = x.value
            agent.init_trait = init_trait
            agent.parents = parents
            agent.srxs = Vector{Tuple{PopulationItx, idType}}()
#            agent.uid = uid#hash(parents, hash(sym, hash(x.value)))
            agent.uid = hash(parents, hash(sym, hash(x.value)))
            agent.consts = consts 
            agent.simulation = nothing
            return agent
        end
    end
end
Base.isequal(a::AgentState, b::AgentState) = isequal(a.uid, b.uid)
getsim(agent::AgentState, t::Float64) = agent.simulation(t)

function get_trait_value(agent::AgentState, t::Float64, pair)::Float64
    pair[1] && return last(agent.consts[pair[2]])
    return @inbounds agent.simulation_interp(t)[pair[2]]
#   return @inbounds agent.simulation(t)[pair[2]]
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

struct SimulationReaction{mType,N1,cType,sType,F1,F2,N2,N3}
    pitx::PopulationItx{mType,N1,cType,sType,F1,F2,N2}
    substrates::NTuple{N3, AgentState}
    sampler::mType
    uid::UInt

    function SimulationReaction(pitx::PopulationItx{mType,N1,cType,sType,F1,F2,N2}, substrates, sampler, rn) where 
        {mType,N1,cType,sType,F1,F2,N2}

        return new{mType,N1,cType,sType,F1,F2,N2,length(substrates)}(pitx, substrates, sampler, hash(substrates, pitx.uid))
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
