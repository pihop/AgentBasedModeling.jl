const ExprValues = Union{Expr, Symbol, Float64, Int}

function process_cnx!(exps, cnxs, pexprs)
    for exp in reverse(exps)  
        exp == nothing && return nothing
        push!(cnxs.args, :(ParameterCnx($(exp)...)))
        push!(pexprs.args, exp.args[1])
    end
end

function process_savesubs!(exps, save)
    for exp in reverse(exps)
        exp == nothing && return nothing
        push!(save.args, :(SaveSubstrateTrait($(exp)...)))
    end
end

function process_saveprods!(exps, save)
    for exp in reverse(exps)
        exp == nothing && return nothing
        push!(save.args, :(SaveProductTrait($(exp)...)))
    end
end

function process_channel(expr, sexpr)
    # Parses reactions, species, and parameters.
    rx = Catalyst.make_reaction(expr.args[3])
    rx
end

function process_transition(exp)
    ex = copy(exp)
    :(TraitTransition($(ex)))
end

function process_vars(ex, vars)
    Catalyst.esc_dollars!(ex)
    var, fnc = ex.args[end].args
    f = copy(fnc)
    fnc_ = quote
        (args...) -> begin
            substitute($f, args...) 
        end
    end
    push!(vars.args, :(AgentBasedModeling.Variable($var, $fnc_)))
end

function process_population_itx(ex)
    itx = :()
    ttran = :(TraitTransition(nothing))
    sampler = :()
    cnxs = :(ParameterCnx[])
    save = :([])
    vars = :(AgentBasedModeling.Variable[])
    name = :(nothing)

    ex = MacroTools.striplines(esc(ex))
    Catalyst.esc_dollars!(ex)

    lines = ex.args
    option_lines = Expr[x for x in lines[1].args if x.head == :macrocall]

    # Get macro options.
    options = Dict(map(arg -> Symbol(String(arg.args[1])[2:end]) => arg,
                       option_lines))
    species_declared = Catalyst.extract_syms(options, :species)
    sexprs = Catalyst.get_sexpr(species_declared, Dict{Symbol, Expr}())

    params_declared = Catalyst.extract_syms(options, :parameters)
    pexprs = Catalyst.get_pexpr(params_declared, Dict{Symbol, Expr}())

    iv = :(@variables $(Catalyst.DEFAULT_IV_SYM))

    lines_dict = Dict() 

    for line in lines[1].args
        line.args[1] == Symbol("@channel") && begin 
            itx = process_channel(line, sexprs)
            lines_dict[:channel] = line
        end
        line.args[1] == Symbol("@transition") && begin 
            ttran = process_transition(line.args[end]) 
            lines_dict[:transition] = line
        end
        line.args[1] == Symbol("@sampler") && begin 
            sampler = line.args[end] 
            lines_dict[:sampler] = line
        end
        line.args[1] == Symbol("@variable") && begin 
            process_vars(line, vars) 
            lines_dict[:variable] = line
        end
        line.args[1] == Symbol("@connections") && begin
            process_cnx!(line.args, cnxs, pexprs)
            lines_dict[:connections] = line
        end
        line.args[1] == Symbol("@savesubstrates") && begin 
            process_savesubs!(line.args, save)
            lines_dict[:savesubstrates] = line
        end
        line.args[1] == Symbol("@saveproducts") && begin 
            process_saveprods!(line.args, save)
            lines_dict[:saveproducts] = line
        end
        line.args[1] == Symbol("@name") && begin 
            name = line.args[end]
        end
    end

    quote
        $iv
        $sexprs
        $pexprs
        PopulationItxDef(TransitionDef($itx, $sampler, $ttran), $sexprs, $pexprs, $cnxs, $vars, saving=$save, name=$name)
    end
end

macro interaction(ex)
    process_population_itx(ex)
end
