# Miscellaneous utility functions.

function indexof(sym, syms)
    return findfirst(isequal(Num(sym)), Num.(syms))
end

function _gen_rate_function(symb_rate, rn::ReactionSystem)
    return Symbolics.build_function(
        symb_rate, 
        tuple(unknowns(rn)...), 
        tuple(ModelingToolkit.parameters(rn)...), 
        ModelingToolkit.get_iv(rn);
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
