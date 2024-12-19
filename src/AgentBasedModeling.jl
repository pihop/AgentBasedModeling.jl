module AgentBasedModeling

    using Catalyst
    using MacroTools
    import JumpProcesses: JumpProblem
    using Symbolics
    using SymbolicUtils
    using ModelingToolkit
    import ModelingToolkit: value
    using SciMLBase 
    using SciMLBase:AbstractTimeseriesSolution,interp_summary
    using JumpProcesses
    using StatsBase
    using ProgressMeter
    using Random
    using CommonSolve 
    using Interpolations
    import Interpolations: scale
    using LinearAlgebra
    using Base.Threads

    const idType = UInt

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

    include("aggregate.jl")
    export FirstReactionMethod, ExtrandeMethod, GillespieMethod, DirectSamplerMethod

    include("simulation.jl")
    export SimulationParameters, simulate

    include("macros.jl")
    export @interaction

end
