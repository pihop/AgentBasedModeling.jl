using Catalyst
using StatsBase

@testset "GillespieMethod -- SI model simulation" begin
    @independent_variables t 
    @abm_variables S(t) I(t)
    @parameters kr ki 

    SAgent = AgentDynamics(EmptyTraitProblem(), ())
    IAgent = AgentDynamics(EmptyTraitProblem(), ())

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

    init_popS = fill(S => (), 99) 
    init_popI = fill(I => (), 1) 
    init_pop = [init_popS; init_popI]

    params = SimulationParameters(
        [ki => 0.1, kr => 1.0], # parameters
        (0, 100), #tspan
        0.1;
        snapshot=[PopulationSnapshot(S), PopulationSnapshot(I)])

    res = simulate(pop, init_pop, params)

    # Theoretical steady state for the SI model is known.
    @test (mean(res.snapshot[:I].u[100:end]) - (1 - 1/(0.1 * 100))*100) < 1
end

