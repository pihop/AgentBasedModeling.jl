### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 355d1e6c-91f6-11ef-2f45-19eb05b5141f
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
		Pkg.PackageSpec(name="OrdinaryDiffEq"),
        Pkg.PackageSpec(name="Catalyst"),
        Pkg.PackageSpec(name="ModelingToolkit"),
		Pkg.PackageSpec(name="Distributions"),
		Pkg.PackageSpec(name="JumpProcesses"),
		Pkg.PackageSpec(name="CairoMakie"),
		Pkg.PackageSpec(name="ColorSchemes"),
		Pkg.PackageSpec(name="StatsBase"),
		Pkg.PackageSpec(name="Integrals"),
		Pkg.PackageSpec(name="Interpolations"),
		Pkg.PackageSpec(name="Graphs"),
		Pkg.PackageSpec(name="MetaGraphsNext"),
		Pkg.PackageSpec(name="NetworkLayout"),
		Pkg.PackageSpec(url="https://github.com/pihop/AgentBasedModeling.jl")
    ])
	using ModelingToolkit
	using OrdinaryDiffEq
	using JumpProcesses
	using LinearAlgebra
	using CairoMakie
	using StatsBase
	using Integrals
	using Graphs
	using MetaGraphsNext
	using NetworkLayout
	using Interpolations
	using ColorSchemes
end

# ╔═╡ e1a5f259-5045-426a-b376-698bff75e96a
using Catalyst, Distributions, AgentBasedModeling

# ╔═╡ 4e3e20d3-d152-4c26-a1f8-a0f4d11cf025
md"""
# AgentBasedModeling.jl demonstration -- Stochastic gene expression coupled to a multi-stage division and replication model
"""

# ╔═╡ 019b50c4-dc51-45eb-bc28-770b8d5c2696
md"""
# Step 1: define cellular agent dynamics
"""

# ╔═╡ 35c0a480-6582-4531-ae36-1135665ee564
begin
	@independent_variables t 
	@abm_variables Δ(t) s(t) p(t) n(t) dna(t) ε(t)
	@parameters α kprod b 
	D = Differential(t)
	
	@register_symbolic Distributions.Geometric(a)
	m = rand(Distributions.Geometric(1/(1 + b*dna)))
	
	CellDynamics = @reaction_network begin
		@species n(t) dna(t) p(t) 
		@variables s(t) Δ(t) ε(t)
		@equations begin
			D(s) ~ $α*s 
	 		D(Δ) ~ $α*s 
            D(ε) ~ $α*s 
		end
    	kprod/1.8, 0 --> $m*p 
	end
end;

# ╔═╡ fee337cd-3f0c-4ccd-a023-7aa4e717c678
Cell = AgentDynamics(CellDynamics, ());

# ╔═╡ 6f00bd54-ffd2-4f41-a533-a35727ba590c
md"""
# Step 2: define multi-stage transition rates, replication and division
"""

# ╔═╡ a5ff7927-087b-43bb-80a9-d9d13c3fe2c5
begin

	γsize(x; μ, cv2) = exp(
    	logpdf(Gamma(1/cv2, μ*cv2), x) - logccdf(Gamma(1/cv2, μ*cv2), x))

	# Adder rule for cell division.
	γdivision(s, Δ, α; μ = 0.5, cv2 = 0.2) = α * s * γsize(Δ; μ=μ, cv2=cv2) 
	@register_symbolic γdivision(s, Δ, α)
	
	# Adder rule for replication.
	γreplication(s, ε, α; μ = 0.5, cv2 = 0.2) = α * s * γsize(ε; μ=μ, cv2=cv2) 
	@register_symbolic γreplication(s, ε, α)


		
	γMS(x; μ, cv2) = (x + 1 >= 1/cv2 ? 1 : 0)

	# multi-step rule for cell division.
	γMS(s, Δ, α, n; μ = 0.5, cv2 = 0.2) = 1/μ * 1/cv2 * α * s * (1. - γMS(n; μ=μ, cv2=cv2))
	@register_symbolic γMS(s, Δ, α, n)
    γdivisionMS(s, Δ, α, n; μ = 0.5, cv2 = 0.2) = 1/μ * 1/cv2 * α * s * γMS(n; μ=μ, cv2=cv2)
	@register_symbolic γdivisionMS(s, Δ, α, n)

	# binomial partitioning
	Distributions.Binomial(p::Float64, β::Real) = Distributions.Binomial(round(Int, p), β)

	@register_symbolic Distributions.Binomial(p::Float64, β)
end;

# ╔═╡ 451f3cf4-ede7-411e-9341-9c4d4b18115b
md"""
Define the intereaction channels and combine everything into a model:
"""

# ╔═╡ a5a789ea-2760-4815-bb71-9390f0db125c
begin
	@abm_variables C(t)
	@parameters Cs CΔ Cp Cn β B L Cdna Cε
	
	transitionMS = @interaction begin
	    @channel γMS($Cs, $CΔ, $α, $Cn), C --> C
	    @sampler ExtrandeMethod($L; boundtype=:increasing)
		@connections (($CΔ => $Δ, $Cs => $s, $Cp => $p, $Cn => $n, $Cε => $ε, $Cdna => $dna),)		
	    @transition (
	        ($Δ => $CΔ, $s => $Cs, $p => $Cp, $n => $Cn + 1.0, $ε => $Cε, $dna => $Cdna + ifelse($Cn == 2.0, 1.0, 0.0), ),)
	    @saveinstate ($C, $s) ($C, $p) ($C, $Δ) ($C, $n)
	    @saveoutstate ($C, $s) ($C, $p) ($C, $Δ) ($C, $n)

		@name "transition"
	end

	divisionMS = @interaction begin
	    @channel γdivisionMS($Cs, $CΔ, $α, $Cn), C --> 2*C
	    @sampler ExtrandeMethod($L; boundtype=:increasing)
		@connections (($CΔ => $Δ, $Cs => $s, $Cp => $p, $Cn => $n, $Cdna => $dna, $Cε => $ε),)		
		@variable $β = rand($Beta(100))
		@variable $B = rand($Binomial($Cp, $β))
	    @transition (
	        ($Δ => 0.0, $s => $β*$Cs, $p => $B, $n => 0.0, $ε => $Cε, $dna => $Cdna/2),
	        ($Δ => 0.0, $s => (1-$β)*$Cs, $p => $Cp - $B, $n => 0.0, $ε => $Cε, $dna => $Cdna/2))

	    @saveinstate ($C, $s) ($C, $p) ($C, $Δ) ($C, $dna)
	    @saveoutstate ($C, $s) ($C, $p) ($C, $Δ) ($C, $dna)
		@name "division"
	end

	model2 = AgentsModel([divisionMS, transitionMS, ], Dict(C => Cell,));
end;

# ╔═╡ 08559fb8-3342-4df5-80d8-4ad280eb472e
md"""
# Step 3: simulation
"""

# ╔═╡ ae0cdda0-fdcf-4eae-b342-d2bf4dd5689d
begin
	# initial condtion
	initial_population = [C => (p => 0.0, s => 0.50, Δ => 0.0, n => 0.0, ε => 0.2, dna =>1. ), ];
	# Simulation timespan.
	timespan = (0.0, 8.0)
	# Simulator timestep.
	Δt = 1.0
	
	params = SimulationParameters(
	    [α => 1.0, kprod => 10.0, b => 6.0, L => 0.01], timespan, Δt; maxpop=1000)

	res2 = simulate(model2, initial_population, params)
end;


# ╔═╡ f777fe28-227a-48a6-9b78-16c66a36ffed
md"""
# Plot results
"""

# ╔═╡ df9500b6-14b4-41f9-a1fb-fd25bf88ee6e
begin
	CairoMakie.activate!()
	colors = ColorSchemes.Hiroshige.colors
	transp = 0.8
	htransp = 0.05
	stairstransp = 0.9
	barstransp = 0.3

	pt_cm = 2.83465
	fig = Figure(size=(300, 80) .* pt_cm; fontsize=8, pt_per_unit=1)
	ax_protein = Axis(fig[1,2]; xlabel="Birth protein distribution", ylabel="Probability density")
	ax_size = Axis(fig[1,1]; xlabel="Birth size distribution", ylabel="Probability density")

    hidedecorations!.(fig.content, ticklabels = false, ticks = false, label = false)
	hidespines!.(fig.content, :t, :r) 
	xlims!(ax_protein, (0, 150))

	function plot_hist!(ax, values; bins, color, label="", weight = 1.0, plotbars = true)
		# Function for plotting the histograms with stair outline.
    	hist_ = normalize(fit(Histogram, values, bins); mode=:pdf)
    	stairs!(ax, collect(midpoints(hist_.edges[1])), hist_.weights .* weight;color=(color, stairstransp), step=:center, label=label)
    	plotbars && barplot!(ax, collect(midpoints(hist_.edges[1])), hist_.weights .* weight; 
        color=(color, barstransp), strokecolor=(color, transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)
	end

	fig

	plot_hist!(ax_size, reduce(vcat, res2.outstates[:Cs_division].u); 
		bins = 0:0.01:1.0, color=colors[1], label = "Agent-based simulation")

    plot_hist!(ax_protein, 
		reduce(vcat, res2.outstates[:Cp_division].u); 
		bins=0:1:120, color=colors[1], label = "Agent-based simulation")

end;

# ╔═╡ 05d96141-f7d0-48e9-9d03-1cc91163fd8b
md"""
Let's compare with analytical computations of effective model (assuming stochastic concentration homeostasis). These follow [(Thomas and Shahrezaei, 2021)](https://royalsocietypublishing.org/doi/epdf/10.1098/rsif.2021.0274). These computations are not part of the package but can be replicated with the following script (hidden by default).

#Thomas P, Shahrezaei V. 2021 Coordination of gene expression noise
#with cell size: analytical results for agent-based models of growing cell populations. J. R. Soc. Interface 18: 20210274. [10.1098/rsif.2021.0274](https://doi.org/10.1098/rsif.2021.0274)
"""

# ╔═╡ b248142d-c524-4e5a-bc25-fbd61aba240b
begin
	#Parametrisation as in the simulation model.
	α_ = 1.0
	kprod_ = 10.0 
	b_ = 6.0
	
	span_ = (1e-4, 1.)
	
	intparams = ( 
	    abstol = 1e-4,
	    reltol = 1e-4)
	
	γ(s, s0) = (1 / (α_ * s)) * γdivision(s, s - s0, α_)
	γint(s, s0) = IntegralProblem((u,p) -> γ(u, s0), (s0, s))
	phi(s, s0) = γ(s, s0) * exp(-solve(γint(s, s0), QuadGKJL(); intparams...).u)

	ddist = Beta(100)
	
	kernelf(s, s0) = IntegralProblem((u,p) -> 2*pdf(ddist, u) * phi(s/u, s0), span_)
	ker(s, s0) = solve(kernelf(s, s0), QuadGKJL(); intparams...).u
	
	function trapz(fx, xstep)
	    out = 0.0
	    fxx = zip(fx, fx[2:end])
	    for f_ in fxx
	        out += middle(f_...) * xstep 
	    end
	    return out
	end
	
	function volterra(ker, span, n)
	    a, b = span
	    h = (b-a)/n
	    x = range(a, stop=b, length=n)
	    
	    Xi = Float64[]
	    Ai = zeros(length(x), length(x))
	    
	    for i in 1:n
	        for j in range(1, n, step=1)
	            Ai[i,j] = h*ker(x[i], x[j])
	        end
	        Ai[1,i] = h*ker(x[1],x[i])
	        Ai[i,i] = h*ker(x[i],x[i])
	    end
	    return Ai
	end
	
	N = 100
	sstep_ = (span_[2]-span_[1])/N
	srange_ = collect(range(span_[1], stop=span_[2], length=N))
	A = volterra(ker, span_, N)
	sn = trapz(Float64.(eigvecs(A)[:,end]), sstep_)
	
	psiA_tree = Float64.(eigvecs(A)[:,end]) ./ srange_
	sn_tree = trapz(psiA_tree, sstep_)
	
	psi = linear_interpolation(
	    range(span_[1], stop=span_[2], length=N), Float64.(eigvecs(A)[:,end]) .* 1/sn, extrapolation_bc=Line())
	psi_tree_ = linear_interpolation(
	    range(span_[1], stop=span_[2], length=N), psiA_tree .* 1/sn_tree, extrapolation_bc=Line())

	# Compute protein distribution for different birth sizes s0.
	function rho(s_, s0_, s0)
	    (1/psi(s0))*(s0/s_)*pdf(ddist, s0/s_)*phi(s_, s0_)*psi(s0_)
	end

	sintspan_ = [span_[1], 2.0]
	# Birth protein counts. Use concentration homeostasis. 
	dist(kprod, b, α, s) = NegativeBinomial(kprod/(α), 1/(1+b*s)) 
	Pi(x, s) = pdf(dist(kprod_, b_, α_, s), x)
	Bpdf(x, x_, θ) = pdf(Binomial(x_, θ), x)
	
	# Integrate
	rhoint(x, x_, s_, s0) = IntegralProblem((u, p) ->  rho(s_, u, s0), sintspan_)
	Pi0int(x, x_, s0) = IntegralProblem(
	    (u,p) -> Bpdf(x, x_, s0/u)*solve(rhoint(x, x_, u, s0), QuadGKJL(); intparams...).u *Pi(x_, u), (s0, sintspan_[end]))
	Pi0(x, x_, s0)  = solve(Pi0int(x, x_, s0), QuadGKJL(); intparams...).u
	
	xs_ = collect(range(0, stop=100, step=5))
	xstep_ = xs_[2] - xs_[1]
	array_dists(x, s0) = map(x_ -> Pi0(x, x_, s0), xs_)
	ss_ = range(sintspan_[1], stop=sintspan_[2], length=20)
	
	mat = [array_dists(x, s) for x in xs_, s in ss_]
	mat_ = hcat([row .* psi_tree_.(ss_) for row in eachrow(mat)]...)'
	mat_ = trapz.(mat_, Ref(xstep_))
	mat_s = trapz.(eachrow(mat_), Ref(sstep_))
	mat_sx = trapz(mat_s, xstep_)
end;

# ╔═╡ 4266ab25-5e95-4241-a257-eb93b6778af1
begin
   # Birth size distribution.
	lines!(ax_size, range(span_[1], stop=span_[2], length=N), psi_tree_.(range(span_[1], stop=span_[2], length=N)); color=colors[10], label="Analytical solution")
	# Birth protein distribution.
	lines!(ax_protein, xs_, mat_s ./ mat_sx; color=colors[10], label="Analytical solution")
	#Legend(fig[2,:], ax_protein; orientation=:horizontal)
	xlims!(ax_protein, (0,  100))
	xlims!(ax_size, (0.2,  1.0))
end;

# ╔═╡ 07d7b398-34b8-4198-9f92-eb58763d04bf
fig

# ╔═╡ Cell order:
# ╠═355d1e6c-91f6-11ef-2f45-19eb05b5141f
# ╠═e1a5f259-5045-426a-b376-698bff75e96a
# ╠═4e3e20d3-d152-4c26-a1f8-a0f4d11cf025
# ╠═019b50c4-dc51-45eb-bc28-770b8d5c2696
# ╠═35c0a480-6582-4531-ae36-1135665ee564
# ╠═fee337cd-3f0c-4ccd-a023-7aa4e717c678
# ╠═6f00bd54-ffd2-4f41-a533-a35727ba590c
# ╠═a5ff7927-087b-43bb-80a9-d9d13c3fe2c5
# ╠═451f3cf4-ede7-411e-9341-9c4d4b18115b
# ╠═a5a789ea-2760-4815-bb71-9390f0db125c
# ╠═08559fb8-3342-4df5-80d8-4ad280eb472e
# ╠═ae0cdda0-fdcf-4eae-b342-d2bf4dd5689d
# ╠═f777fe28-227a-48a6-9b78-16c66a36ffed
# ╠═df9500b6-14b4-41f9-a1fb-fd25bf88ee6e
# ╠═05d96141-f7d0-48e9-9d03-1cc91163fd8b
# ╟─b248142d-c524-4e5a-bc25-fbd61aba240b
# ╠═4266ab25-5e95-4241-a257-eb93b6778af1
# ╠═07d7b398-34b8-4198-9f92-eb58763d04bf
