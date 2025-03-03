abstract type AbstractSaveTrait end

mutable struct SimulationResults
    instates::Dict
    outstates::Dict
    snapshot::Dict
    agents
    final_pop
    interactions
    tend

    function SimulationResults(model; snapshot)
        ins = Dict()
        outs = Dict()
        sshot = Dict()

        return new(ins, outs, sshot, nothing, nothing, [])
    end
end

function Base.show(io::IO, ::MIME"text/plain", results::SimulationResults)
    print(io, "Simulation results")
end


struct SaveInStateTrait <: AbstractSaveTrait
    agent
    trait
end

struct SaveOutStateTrait <: AbstractSaveTrait
    agent
    trait
end

struct TraitSnapshot <: AbstractSaveTrait
    agent
    trait
end

struct PopulationSnapshot <: AbstractSaveTrait
    agent
end

function save_trait_name(strait::AbstractSaveTrait)
    Symbol(strait.agent.val.f, strait.trait.val.f)
end

function save_trait_name(strait::PopulationSnapshot)
    Symbol(strait.agent.val.f)
end

struct TraitValue{V,T}
    value::V
    time::T
    id::Int
end

struct TraitValueRx{V,T,rxType}
    sym::Num
    value::V
    time::T
    id::Int
    rx::rxType
end

function Base.show(io::IO, ::MIME"text/plain", results::TraitValueRx)
    println(io, "[$(results.sym), $(results.value), $(results.time), $(results.id), $(results.rx.rx)]")
end

struct Snapshot{tType, vType}
    time::tType
    values::vType
end

struct SnapshotSolution{T,N,uType,tType,IType} <: AbstractTimeseriesSolution{T,N,uType}
    names::Vector{Symbol}
    u::uType
    t::tType
    interp::IType
    retcode::ReturnCode.T
end

function SnapshotSolution{T, N}(names, u, t, interp, retcode) where {T, N}
    return SnapshotSolution{T, N, typeof(u), typeof(t), typeof(interp)}(names, u, t, interp, retcode)
end

function (sol::SnapshotSolution)(t::Number)
    return sol.interp(t)
end

function build_snapshot_solution(snapshot; names)
    t = getfield.(snapshot[names[1]], :time)
    snaps = zip([getfield.(snapshot[name], :values) for name in names]...)
    us = [[s...] for s in snaps] 
    T = eltype(eltype(us))
    N = length((size(us[1])..., length(us)))
    Interpolations.deduplicate_knots!(t)
    interp = Interpolations.linear_interpolation(t, us)  
    retcode = ReturnCode.Success
    return SnapshotSolution{T,N}(names, us, t, interp, retcode)
end

SciMLBase.interp_summary(::T) where T <: Interpolations.Extrapolation = "Linear Interpolation"

function lineage(res, cell)
    lin = [] 
    isnothing(cell.parents) && return [cell, ]
    cell_ = cell.parents[1]
    while true 
        savecell_ = res.agents[cell_.sym][cell_.uid]
        push!(lin, savecell_)
        isnothing(savecell_.parents) && return reverse(lin)
        cell_ = savecell_.parents[1]
    end
    return reverse(lin)
end
