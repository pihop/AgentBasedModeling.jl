module AgentBasedModeling

    using Catalyst
    using MacroTools
#    using DiffEqBase
    import JumpProcesses: JumpProblem
    using Symbolics
    using SymbolicUtils
#    using OrderedCollections
    using ModelingToolkit
    import ModelingToolkit: value
#    using DataStructures
#    MT = ModelingToolkit
#    using Distributions 
    using SciMLBase 
    using SciMLBase:AbstractTimeseriesSolution,interp_summary
    using JumpProcesses
#    using RuntimeGeneratedFunctions
    using StatsBase
    using ProgressMeter
    using Random
#    using FunctionWrappers
#    import FunctionWrappers: FunctionWrapper
    using CommonSolve 
    using Interpolations
    import Interpolations: scale
    using LinearAlgebra
#    using InteractiveUtils
#    using BenchmarkTools
    using Base.Threads
    
#    const idType = UUID
    const idType = UInt

#    RuntimeGeneratedFunctions.init(@__MODULE__)
    const DEFAULT_RNG = Random.default_rng()

    include("utils.jl")
    include("results.jl")
    export population_counts, SaveSubstrateTrait, SaveProductTrait, TraitSnapshot, PopulationSnapshot

    include("models.jl")
    export AgentDynamics, AgentState, PopulationModelDef, ParameterCnx, AgeConnection, Variable, TraitTransition, PopulationItx
    export TransitionDef
    export EmptyTraitProblem
    export HybridSDEDynamics

    include("sampling.jl")
    export ThinningSampler, DirectSampler

    include("aggregate.jl")
    export FirstReactionMethod, ExtrandeMethod, GillespieMethod

    include("simulation.jl")
    export SimulationParameters, simulate

    include("macros.jl")
    export @interaction

end
