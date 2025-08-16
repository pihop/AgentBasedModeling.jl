
function chemostat_rate(C_, N_, r_)
    C_ > N_ && return r_
    return 0.0    
end

@register_symbolic chemostat_rate(C, N, r)

chemostat = @interaction begin
    @channel chemostat_rate($C, $Nmax, $r), $C --> 0 
    @sampler ExtrandeMethod($chemostat_rate($C, $Nmax, $r), Inf;)
end

function runchemostat(model, init_pop, params, N, M)
    birth_pdist = [] 
    div_tdist = [] 
    init_pop_ = init_pop
    res_ = nothing
    
    for i in 1:M
        println(i)
        res = simulate(population_model, init_pop_, simulation_params; remember_all_agents=true)
        push!(birth_pdist, getfield.(res.prods[:CP], :value)...)
        push!(div_tdist, getfield.(res.prods[:Cs], :value)...)
        tend = res.tend
        agents = rand(res.final_pop[C], N)
        init_pop_ = repeat([C,], N) .=> 
            [(P => a[2].simulation(tend; idxs=P), 
              s => a[2].simulation(tend; idxs=s),
              s0 => a[2].simulation(a[2].btime; idxs=s),
              τ => a[2].simulation(tend; idxs=τ)) for a in agents ]
        res_ = res
    end
    return res_, birth_pdist, div_tdist
end


