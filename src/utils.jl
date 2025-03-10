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

    for eq in eqs
        isequal(eq.rhs, 0.0) && continue
        if isequal(eq.lhs, 0)
            vs = get_variables(eq)
            display(vs)
#            display(ModelingToolkit.isparameter(vs[2]))
#            display(ModelingToolkit.getvariabletype(eq.rhs))
        else
            push!(vars, eq.lhs.arguments)
        end
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
        (rx.rate isa Symbolic) && get_variables!(rxvars, rx.rate)
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
        (rx.rate isa Symbolic) && get_variables!(rxvars, rx.rate)

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
