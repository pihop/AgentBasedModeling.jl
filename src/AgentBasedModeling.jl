module AgentBasedModeling

    using Catalyst
    import Catalyst: get_rxs, drop_dynamics, get_depgraph, dfs_mark!, makemajump
    import Catalyst: value, get_variables!, get_iv, get_unknowns
    using MacroTools

    using JumpProcesses
    import JumpProcesses: JumpProblem, extend_problem
    using Symbolics
    import Symbolics: Symbolic
    using SymbolicUtils
    using OrdinaryDiffEq
    using StochasticDiffEq

#    using ModelingToolkit
    MT = ModelingToolkit
    using SciMLBase 
    using SciMLBase:AbstractTimeseriesSolution,interp_summary
    using RecursiveArrayTools

    using StatsBase
    using ProgressMeter
    using Random
    using CommonSolve 
    using Interpolations
    using Accessors

    using Graphs
    using MetaGraphsNext

    using Base.Threads
    
    const idType = UInt

    const DEFAULT_RNG = Random.default_rng()

    include("utils.jl")
    include("results.jl")
    export population_counts, SaveSubstrateTrait, SaveProductTrait, StateSnapshot, PopulationSnapshot
    export construct_interaction_graph

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
    export SimulationParameters, simulate, simulate_step!

    include("macros.jl")
    export @interaction
    export @abm_variables
end
