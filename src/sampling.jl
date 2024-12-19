struct BadRateBound <: Exception end
Base.showerror(io::IO, e::BadRateBound) = print(io, "Defined rate bound invalid in the time interval.")

struct GillespieMethod end
get_λmax(s::GillespieMethod) = 0.0 
get_L(s::GillespieMethod) = Inf 

function sample_first_arrival(ratef, pop, pvec, pmod, subsrules, subs, state, tspan, sampler::GillespieMethod, model; ratemax=0, Lf=nothing)
    pstate!(pmod, pvec, subsrules, model, subs, state, tspan[1])
    λ = ratef(state.pop_state, pvec, tspan[1])
    proposet = tspan[1] + randexp() / λ 
    return proposet
end

struct ExtrandeMethod
    λmax::Num
    L::Num
    trait_indep::Bool
    pop_indep::Bool
    
    function ExtrandeMethod(λmax, L; trait_indep=false, pop_indep=false)
        new(λmax, L, trait_indep, pop_indep)
    end
end

Base.show(io::IO, sampler::ExtrandeMethod) = print(io, "Extrande method")

get_λmax(s::ExtrandeMethod) = s.λmax
get_L(s::ExtrandeMethod) = s.L

struct FirstReactionMethod
    λmax::Num
    L::Num
    trait_indep::Bool
    pop_indep::Bool
    
    function FirstReactionMethod(λmax, L; trait_indep=false, pop_indep=false)
        new(λmax, L, trait_indep, pop_indep)
    end
end

Base.show(io::IO, sampler::FirstReactionMethod) = print(io, "First reaction method")

get_λmax(s::FirstReactionMethod) = s.λmax
get_L(s::FirstReactionMethod) = s.L

function sample_first_arrival(ratef, pop, pvec, pmod, subsrules, subs, state, tspan, sampler::FirstReactionMethod, model; ratemax, Lf)
    proposet = tspan[1]
    pstate!(pmod, pvec, subsrules, model, subs, state, proposet)

    while true
        last_prop = proposet

        λmax = ratemax(state.pop_state, pvec, proposet)
        looka = Lf(state.pop_state, pvec, proposet)

        proposet += randexp() / λmax 

        proposet > last_prop + looka && begin
            proposet = last_prop + looka
            proposet ≥ tspan[end] && return Inf
            pstate!(pmod, pvec, subsrules, model, subs, state, proposet)
            continue
        end

        proposet ≥ tspan[end] && return Inf

        U = rand()
        pstate!(pmod, pvec, subsrules, model, subs, state, proposet)
        λt = ratef(state.pop_state, pvec, proposet)

        if λt > λmax  
            @error "Bound evaluated as $(λmax) with rate evaluated as $(λt). $(ratef)"
            throw(BadRateBound) 
        elseif (U*λmax ≤ λt) 
            return proposet
        end
    end
end

struct DirectSamplerMethod
#    L::Num
#    
#    function DirectSamplerMethod(L)
#        new(L)
#    end
end

Base.show(io::IO, sampler::DirectSamplerMethod) = print(io, "First reaction method")

get_λmax(s::DirectSamplerMethod) = 0.0 
get_L(s::DirectSamplerMethod) = Inf 

function sample_first_arrival(ratef, pop, pvec, pmod, subsrules, subs, state, tspan, sampler::DirectSamplerMethod, model; ratemax, Lf)
    proposet = tspan[1]
    pstate!(pmod, pvec, subsrules, model, subs, state, proposet)
    proposet += ratef(state.pop_state, pvec, proposet)
    return proposet
end

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
