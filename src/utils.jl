# Miscellaneous utility functions.

function indexof(sym, syms)
    return findfirst(isequal(Num(sym)), Num.(syms))
end

function _gen_rate_function(symb_rate, rn::ReactionSystem)
    return Symbolics.build_function(
        symb_rate, 
        tuple(unknowns(rn)...), 
        tuple(MT.parameters(rn)...), 
        get_iv(rn);
#        force_SA = true,
#        conv = ModelingToolkit.states_to_sym(states(rn)),
#        parallel=MultithreadedForm(),
        expression=Val{false})
end

function trait_key(type, trait)
    return string(type.val.f, trait.f)  
end

isnum(x) = typeof(x) == Num

#using Base.Threads

const print_lock = SpinLock()
const prints_pending = Vector{String}()
function tprintln(str)
	tid= Threads.threadid()
	str = "[Thread $tid]: " * string(str)
	lock(print_lock) do
		push!(prints_pending, str)
		if tid == 1 # Only first thread is allows to print
			println.(prints_pending)
			empty!(prints_pending)
		end
	end
end

# Stuff below will be deleted once hybrid systems are supported by Catalyst.jl. 
function get_time_dependent_vars(eqs)
    vars = Set()
    discvars = Set()

    diffeqs = filter(x -> MT.isdiffeq(x), eqs)
    nondiffeqs = filter(x -> !MT.isdiffeq(x), eqs)

    for eq in diffeqs
        # If rhs 0 then we only have discrete jumps corresponding to the variable.
        vs = get_variables(eq)
        isequal(eq.rhs, 0.0) && begin
            push!(discvars, vs...)
            continue
        end
        vs_topush = filter(x -> !MT.isparameter(x), vs)
        push!(vars, vs_topush...)

    end

    for eq in nondiffeqs
        vs = get_variables(eq)
        # Are any of the variables continuous?
        anyvars = any(x -> !in(x, vars), vs)
        vs_topush = filter(x -> !MT.isparameter(x), vs)
        push!(vars, vs_topush)
    end

    vars
end

function assemble_hybrid_jumps(rs; combinatoric_ratelaws = true)
    meqs = MassActionJump[]
    ceqs = ConstantRateJump[]
    veqs = VariableRateJump[]
    unknownset = Set(get_unknowns(rs))

    rxvars = []

    isempty(get_rxs(rs)) &&
        error("Must give at least one reaction before constructing a JumpSystem.")

    eqs = setdiff(equations(rs), reactions(rs))
    # Get continuous time variables. 
    cont_time_vars = Set(reduce(vcat, get_time_dependent_vars(eqs), init=[]))

    # first we determine vrjs with an explicit time-dependent rate
    rxs = get_rxs(rs)
    isvrjvec = falses(length(rxs))
    havevrjs = false
    for (i, rx) in enumerate(rxs)
        empty!(rxvars)
        (rx.rate isa SymbolicUtils.BasicSymbolic) && get_variables!(rxvars, rx.rate)
        @inbounds for rxvar in rxvars
            if (isequal(rxvar, get_iv(rs)) | in(rxvar, cont_time_vars))
                isvrjvec[i] = true
                havevrjs = true
                break
            end
        end
    end

    # now we determine vrj's that depend on species modified by a previous vrj
    if havevrjs
        depgraph = get_depgraph_temp(rs)
        visited = falses(length(isvrjvec))
        for (i, isvrj) in enumerate(isvrjvec)
            if isvrj && !visited[i]
                # dfs from the vrj node to propagate vrj classification
                dfs_mark!(isvrjvec, visited, depgraph, i)
            end
        end
    end

    for (i, rx) in enumerate(rxs)
        empty!(rxvars)
        (rx.rate isa SymbolicUtils.BasicSymbolic) && get_variables!(rxvars, rx.rate)

        isvrj = isvrjvec[i]
        if (!isvrj) && ismassaction(rx, rs; rxvars, haveivdep = false, unknownset)
            push!(meqs, makemajump(rx; combinatoric_ratelaw = combinatoric_ratelaws))
        else
            rl = jumpratelaw(rx; combinatoric_ratelaw = combinatoric_ratelaws)
            affect = Vector{Equation}()
            for (spec, stoich) in rx.netstoich
                # don't change species that are constant or BCs
                (!drop_dynamics(spec)) && push!(affect, spec ~ spec + stoich)
            end
            if isvrj
                push!(veqs, VariableRateJump(rl, affect))
            else
                push!(ceqs, ConstantRateJump(rl, affect))
            end
        end
    end
    vcat(meqs, ceqs, veqs)
end

function get_depgraph_temp(rs)
    jdeps = asgraph(rs)
    vdeps = variable_dependencies(rs)
    eqs = reactions(rs)
    jdeps = asgraph(rs; eqs)
    vdeps = variable_dependencies(rs; eqs)
    eqeq_dependencies(jdeps, vdeps).fadjlist
end

function extend_problem(prob::DiffEqBase.SDEProblem, jumps; rng = DEFAULT_RNG)
    # Modify the extend problem to avoid remake. Remake for SDEProblems seems broken in JumpProceses.
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, p, t)
                _f(du.u, u.u, p, t)
                JumpProcesses.update_jumps!(du, u, p, t, length(u.u), jumps...)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, p, t)
                du = ExtendedJumpArray(_f(u.u, p, t), u.jump_u)
                JumpProcesses.update_jumps!(du, u, p, t, length(u.u), jumps...)
                return du
            end
        end
    end

    if prob.noise_rate_prototype === nothing
        jump_g = function (du, u, p, t)
            prob.g(du.u, u.u, p, t)
        end
    else
        jump_g = function (du, u, p, t)
            prob.g(du, u.u, p, t)
        end
    end

    u0 = JumpProcesses.extend_u0(prob, length(jumps), rng)
    f = SDEFunction{isinplace(prob)}(jump_f, jump_g; sys = prob.f.sys,
        observed = prob.f.observed)
    SDEProblem(f, prob.g, u0, prob.tspan, prob.p; prob.kwargs...)
end
