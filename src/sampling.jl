struct BadRateBound <: Exception end
Base.showerror(io::IO, e::BadRateBound) = print(io, "Defined rate bound invalid in the time interval.")

struct GillespieMethod end
get_λmax(s::GillespieMethod) = 0.0 
get_L(s::GillespieMethod) = Inf 

function sample_first_arrival(ratef, pop, pvec, pmod, subsrules, subs, state, tspan, sampler::GillespieMethod, model; ratemax=0, Lf=nothing)
    pstate!(pmod, pvec, subsrules, subs, tspan[1])
    λ = ratef(state.pop_state, pvec, tspan[1])
    proposet = tspan[1] + randexp() / λ 
    return proposet
end

struct UnknownBound end
struct IncreasingRate end
struct DecreasingRate end
struct SpecifiedBound end

struct ExtrandeMethod{F, M}
    λmax::F
    L::Num
    trait_indep::Bool
    pop_indep::Bool
end

function ExtrandeMethod(L; trait_indep=false, pop_indep=false, boundtype=:unknown)
    if boundtype == :unknown
        return ExtrandeMethod{typeof(nothing),typeof(UnknownBound())}(nothing, L, trait_indep, pop_indep)
    end

    if boundtype == :increasing
        return ExtrandeMethod{typeof(nothing),typeof(IncreasingRate())}(nothing, L, trait_indep, pop_indep)
    end

    if boundtype == :decreasing
        return ExtrandeMethod{typeof(nothing),typeof(DecreasingBound())}(nothing, L, trait_indep, pop_indep)
    end

    error("Bound type not recognized. Valid options are :unknown, :increasing and :decreasing")
end

function ExtrandeMethod(λmax, L; trait_indep=false, pop_indep=false)
    ExtrandeMethod{typeof(λmax),typeof(SpecifiedBound())}(λmax, L, trait_indep, pop_indep)
end

Base.show(io::IO, sampler::ExtrandeMethod) = print(io, "Extrande method")

get_λmax(s::ExtrandeMethod) = s.λmax
get_L(s::ExtrandeMethod) = s.L

struct FirstReactionMethod{F, M}
    λmax::F
    L::Num
    trait_indep::Bool
    pop_indep::Bool
end

function FirstReactionMethod(L; trait_indep=false, pop_indep=false, boundtype=:unknown)
    if boundtype == :unknown
        return FirstReactionMethod{typeof(nothing),typeof(UnknownBound())}(nothing, L, trait_indep, pop_indep)
    end

    if boundtype == :increasing
        return FirstReactionMethod{typeof(nothing),typeof(IncreasingRate())}(nothing, L, trait_indep, pop_indep)
    end

    if boundtype == :decreasing
        return FirstReactionMethod{typeof(nothing),typeof(DecreasingBound())}(nothing, L, trait_indep, pop_indep)
    end

    error("Bound type not recognized. Valid options are :unknown, :increasing and :decreasing")
end

function FirstReactionMethod(λmax, L; trait_indep=false, pop_indep=false)
    FirstReactionMethod{typeof(λmax), typeof(SpecifiedBound())}(λmax, L, trait_indep, pop_indep)
end

Base.show(io::IO, sampler::FirstReactionMethod) = print(io, "First reaction method")

get_λmax(s::FirstReactionMethod) = s.λmax
get_L(s::FirstReactionMethod) = s.L

function sample_first_arrival(ratef, pop, pvec, pmod, subsrules, subs, state, tspan, sampler::FirstReactionMethod, model; ratemax, Lf)
    proposet = tspan[1]
    pstate!(pmod, pvec, subsrules, subs, proposet)

    while true
        last_prop = proposet

        # Evaluate bound and lookahead at last proposed time.
        λmax = ratemax(state.pop_state, pvec, last_prop)
        looka = Lf(state.pop_state, pvec, last_prop)

        # Propose a new time.
        proposet += randexp() / λmax 

        proposet > last_prop + looka && begin
            # If outside the currently valid interval.
            proposet = last_prop + looka
            # If the proposal outside the simulated timespan return Inf (no interaction).
            proposet ≥ tspan[end] && return Inf
            # Else set state to the new time and keep sampling.
            pstate!(pmod, pvec, subsrules, subs, proposet)
            continue
        end

        proposet ≥ tspan[end] && return Inf

        U = rand()
        pstate!(pmod, pvec, subsrules, subs, proposet)
        λt = ratef(state.pop_state, pvec, proposet)

        # Catch misspecification of bounds.
        λt > λmax && begin
            @error "Bound evaluated as $(λmax) at time $(proposet) with rate evaluated as $(λt). $(ratef)"
            throw(BadRateBound) 
        end

        # Acceptance criterion.
        if (U*λmax ≤ λt) 
            return proposet
        end
    end
end

#struct DirectSampler
#    L::Num
#    
#    function DirectSampler(L)
#        new(L)
#    end
#end
#
#get_λmax(s::DirectSampler) = 0.0 
#get_L(s::DirectSampler) = s.L
#
#function process_interaction!(inter::PopulationItx{TransitionDef{DirectSampler}}, rn)
#    inter.rx.method.propf = _gen_rate_function(inter.rx.method.prop_sym, rn)
#    inter.rx.method.lfn = _gen_rate_function(inter.rx.method.lfn_sym, rn)
#end
#
#@inline function sample_first_arrival(ratef, pop, pvec, pmod, subsrules, subs, state, tspan, sampler::DirectSampler, model; ratemax, Lf)
#    proposet = tspan[1]
#    last_prop = tspan[1]
#    pstate!(pmod, pvec, subsrules, model, subs, state, last_prop)
#    proposet += ratef(state.pop_state, pvec, last_prop)
#    return proposet
#end
#
#mutable struct DirectSampler
#    propf::Function
#    prop_sym::Num
#    proposet
#    bt
#    λ::Function
#    tspan::Tuple{Float64, Float64}
#    lfn_sym::Num
#    lfn::Function
#    L::Function
#
#    function DirectSampler(propf, lfn)
#        sampler = new()
#        sampler.prop_sym = propf
#        sampler.lfn_sym = lfn
#        sampler.tspan = (0.0, 0.0)
#        return sampler
#    end
#end
#function init_sampler(sampler::DirectSampler, tspan)
#    sampler_ = sampler
#    sampler_.proposet = nothing 
#    sampler_.tspan = tspan
#    sampler_.bt = tspan[1]
#    return sampler_
#end
#
#function update_sampler!(sampler::DirectSampler, pop_state, pstate, tspan)
#    sampler.tspan = (tspan[1], tspan[end]) 
#    λ(t) = sampler.propf(pop_state, pstate(t), t) 
#    sampler.λ = t -> λ(t)
#    L(t) = sampler.lfn(pop_state, pstate(t), t)
#    sampler.L = t -> L(t)
#end
#
#function propose_next!(sampler::DirectSampler)
#    sample = sampler.λ(sampler.proposet)
#    if isnothing(sample)
#        sampler.proposet = nothing
#    else 
#        sampler.proposet = sampler.proposet + sample 
#    end
#end


#function sample_first_arrival!(sampler::DirectSampler)
#    sampler.proposet = sampler.tspan[1]
#    last_prop = sampler.tspan[1]
#    while true
#        propose_next!(sampler)
#        if isnothing(sampler.proposet)
#            sampler.proposet = last_prop + sampler.L(last_prop) + 1e-12 
#            last_prop = sampler.proposet
#        elseif sampler.proposet > sampler.tspan[end]
#            sampler.proposet = Inf
#            return Inf
#        elseif sampler.proposet ≤ sampler.tspan[end]
#            return sampler.proposet
#        end
#
#        if last_prop > sampler.tspan[end]
#            sampler.proposet = Inf 
#            return Inf
#        end
#    end
#end
