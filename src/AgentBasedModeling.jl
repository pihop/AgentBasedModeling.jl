module AgentBasedModeling

    using Catalyst
    import Catalyst: get_rxs, drop_dynamics, get_depgraph, dfs_mark!, makemajump
    using MacroTools

    import JumpProcesses: JumpProblem
    using Symbolics
    import Symbolics: Symbolic
    using SymbolicUtils
    using OrdinaryDiffEq

#    using ModelingToolkit
    import Catalyst: value, get_variables!, get_iv, get_unknowns
    MT = Catalyst.ModelingToolkit
    using SciMLBase 
    using SciMLBase:AbstractTimeseriesSolution,interp_summary
    using RecursiveArrayTools
    using JumpProcesses
    import JumpProcesses: extend_problem

    using StatsBase
    using ProgressMeter
    using Random
    using CommonSolve 
    using Interpolations
    import Interpolations: scale

    using InteractiveUtils
    using Base.Threads
    
    const idType = UInt

    const DEFAULT_RNG = Random.default_rng()

    include("utils.jl")
    include("results.jl")
    export population_counts, SaveSubstrateTrait, SaveProductTrait, StateSnapshot, PopulationSnapshot

    include("models.jl")
    export AgentDynamics, AgentState, AgentsModel, ParameterCnx, AgeConnection, Variable, TraitTransition, PopulationItx
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
    export @abm_variables
end
