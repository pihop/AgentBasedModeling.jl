using ModelingToolkit
using Catalyst
using StatsBase

@independent_variables t 
@abm_variables S(t) I(t)
@parameters kr ki 

SAgent = AgentDynamics(EmptyTraitProblem(), ())
IAgent = AgentDynamics(EmptyTraitProblem(), ())

init_popS = fill(S => (), 99) 
init_popI = fill(I => (), 1) 
init_pop = [init_popS; init_popI]

params = SimulationParameters(
    [ki => 0.1, kr => 1.0], # parameters
    (0, 100), #tspan
    0.1;
    snapshot=[PopulationSnapshot(S), PopulationSnapshot(I)])

@testset "GillespieMethod -- SI model simulation constant rate" begin
    infection = @interaction begin
        @channel ki, $S + $I --> 2*$I
        @sampler GillespieMethod()
        @transition ((),())
    end

    recovery = @interaction begin
        @channel kr, $I --> $S
        @sampler GillespieMethod()
        @transition ((),())
    end

    pop = AgentsModel([infection, recovery], Dict(S => SAgent, I => IAgent))
    res = simulate(pop, init_pop, params)

    # Theoretical steady state for the SI model is known.
    @test (mean(res.snapshot[:I].u[100:end]) - (1 - 1/(0.1 * 100))*100) < 1
end

@testset "ExtrandeMethod -- SI model simulation constant rate" begin
    infection = @interaction begin
        @channel ki, $S + $I --> 2*$I
        @sampler ExtrandeMethod(1.0)
        @transition ((),())
    end

    recovery = @interaction begin
        @channel kr, $I --> $S
        @sampler ExtrandeMethod(1.0)
        @transition ((),())
    end

    pop = AgentsModel([infection, recovery], Dict(S => SAgent, I => IAgent))
    res = simulate(pop, init_pop, params)

    # Theoretical steady state for the SI model is known.
    @test (mean(res.snapshot[:I].u[100:end]) - (1 - 1/(0.1 * 100))*100) < 1
end

@testset "FirstReactionMethod -- SI model simulation constant rate" begin
    infection = @interaction begin
        @channel ki, $S + $I --> 2*$I
        @sampler FirstReactionMethod(1.0)
        @transition ((),())
    end

    recovery = @interaction begin
        @channel kr, $I --> $S
        @sampler FirstReactionMethod(1.0)
        @transition ((),())
    end

    pop = AgentsModel([infection, recovery], Dict(S => SAgent, I => IAgent))
    res = simulate(pop, init_pop, params)

    # Theoretical steady state for the SI model is known.
    @test (mean(res.snapshot[:I].u[100:end]) - (1 - 1/(0.1 * 100))*100) < 1
end
