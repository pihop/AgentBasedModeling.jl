using Catalyst

@independent_variables t 
@species C(t) P(t) s(t)
@parameters α s0 θ[1:2] Cs Cs0
D = Differential(t)

@named cont_dynamics = System([D(s) ~ α*s, ], t)

disc_dynamics = @reaction_network begin
    @species s(t)
    kprod, 0 --> P 
end

hybrid_dynamics = @reaction_network begin
    @variables s(t)
    @species P(t)
    @parameters α
    @equations D(s) ~ α*s
    kprod, 0 --> P 
end

@testset "make_trait_problem builds ODEProblem for ODE system" begin
    dyn = AgentDynamics(cont_dynamics, (s0,))
    trait = AgentBasedModeling.make_trait_problem(C, dyn, (0.0, 1.0), [α => 0.1])
    @test trait isa AgentBasedModeling.Trait
    @test trait.problem isa ODEProblem
end

@testset "AgentDynamics constructor doesn't error" begin
    @test AgentDynamics(cont_dynamics, (s0, )) isa AgentDynamics
    @test AgentDynamics(disc_dynamics, (s0, )) isa AgentDynamics
    @test AgentDynamics(hybrid_dynamics, (s0, )) isa AgentDynamics 
end

@testset "Interaction defintion doesn't error" begin
    @test begin 
        itx = @interaction begin
            @channel 1.0, $C --> 2*$C
            @sampler ExtrandeMethod(1.0, 0.1)
            @variable $θ = [0.4, 0.6] 
            @transition (
                ($s => $θ[1]*$Cs, $s0 => $θ[1]*$Cs),
                ($s => $θ[2]*$Cs, $s0 => $θ[2]*$Cs))
            @connections (($Cs => $s, $Cs0 => $s0),)
            @saveinstates ($C, $s)
            @saveoutstates ($C, $s)
        end 
        itx isa AgentBasedModeling.PopulationItxDef
    end

    @test begin 
        itx = @interaction begin
            @channel 1.0, $C --> 2*$C
            @sampler FirstReactionMethod(1.0, 0.1)
            @variable $θ = [0.4, 0.6] 
            @transition (
                ($s => $θ[1]*$Cs, $s0 => $θ[1]*$Cs),
                ($s => $θ[2]*$Cs, $s0 => $θ[2]*$Cs))
            @connections (($Cs => $s, $Cs0 => $s0),)
            @saveinstates ($C, $s)
            @saveoutstates ($C, $s)
        end 
        itx isa AgentBasedModeling.PopulationItxDef
    end

    @test begin 
        itx = @interaction begin
            @channel 1.0, $C --> 2*$C
            @sampler GillespieMethod()
            @variable $θ = [0.4, 0.6] 
            @transition (
                ($s => $θ[1]*$Cs, $s0 => $θ[1]*$Cs),
                ($s => $θ[2]*$Cs, $s0 => $θ[2]*$Cs))
            @connections (($Cs => $s, $Cs0 => $s0),)
            @saveinstates ($C, $s)
            @saveoutstates ($C, $s)
        end 
        itx isa AgentBasedModeling.PopulationItxDef
    end
end
