using OrdinaryDiffEq
using OrdinaryDiffEq
using AgentBasedModeling
using Catalyst
using CairoMakie
using Distributions
using StatProfilerHTML
using JumpProcesses
using JLD2
using StatsBase
using LinearAlgebra

using ColorSchemes
colors = ColorSchemes.Hiroshige.colors
transp = 0.8
htransp = 0.05
stairstransp = 0.9
barstransp = 0.3

@independent_variables t 
@species C(t) P(t) τ(t) s(t)
@parameters Cτ Cs Cs0 CP Cθ s0 α θ[1:4] kprod K b μ cv L Nmax r
D = Differential(t)

@register_symbolic Distributions.Geometric(a)

m = rand(Distributions.Geometric(1/(1 + b*s))) 

disc_dynamics = @reaction_network begin
    @species s(t)
    kprod, 0 --> $m*P 
end

@named cont_dynamics = ODESystem([D(s) ~ α*s, D(τ) ~ 1.0, D(P) ~ 0.0], t)
Cell = AgentDynamics((cont_dynamics, disc_dynamics), (s0, ))

gammahaz(μ,cv,x) = exp(logpdf(Gamma(1/cv, μ*cv), x) - logccdf(Gamma(1/μ, μ*cv), x))

# Adder rule for cell division.
γdiv(μ, cv, s, s0, α) = α * s * gammahaz(μ, cv, s - s0)
@register_symbolic γdiv(μ, cv, s, s0, α)

function γdivbound(μ, cv, τ, s0, L, α)
   s0*exp(α*(τ+L)) * α * gammahaz(μ, cv, s0*exp(α*(τ+L)) - s0)
end
@register_symbolic γdivbound(μ, cv, τ, s0, L, α)

function sample_beta()
    rand(Beta(100))
end

function partition_cell(x)
    p = sample_beta()
    x_ = floor(x)
    x1 = rand(Binomial(x_, p))
    x2 = x_ - x1
    return [p, 1-p, x1, x2]
end

@register_symbolic binomial_part(a)
@register_symbolic partition_cell(a)

division = @interaction begin
    @channel γdiv($μ, $cv, $Cs, $Cs0, $α), $C --> 2*$C
    @sampler ExtrandeMethod($γdivbound($μ, $cv, $Cτ, $Cs0, $L, $α), $L)
    @variable $θ = $partition_cell($CP)
    @transition (
        ($τ => 0.0, $s => $θ[1]*$Cs, $s0 => $θ[1]*$Cs, $P => $θ[3]),
        ($τ => 0.0, $s => $θ[2]*$Cs, $s0 => $θ[2]*$Cs, $P => $θ[4]))
    @connections ($Cτ, $C, $τ) ($Cs, $C, $s) ($Cs0, $C, $s0) ($CP, $C, $P)
    @savesubstrates ($C, $s)
    @savesubstrates ($C, $P)
    @saveproducts ($C, $s)
    @saveproducts ($C, $P)
end

# Put them together.
population_model = PopulationModelDef([division, ], Dict(C => Cell,))
# Initial population state. Starting with a single cell at age 0.
init_pop = repeat([C,], 1) .=> repeat([(P => 30.0, τ => 0.0, s => 0.50, s0 => 0.50),], 1)
tspan_pop = (0.0, 20.0)
# Simulator timestep.
Δt = 1.0

simulation_params = SimulationParameters(
    [α => 1.0, kprod => 10.0, b => 6.0, μ => 1.0, cv => 0.2, L => 0.1], 
    tspan_pop, 
    Δt, 
    Rodas5(); 
    snapshot=[PopulationSnapshot(C),], 
    maxpop=10000)

res = simulate(population_model, init_pop, simulation_params; remember_all_agents=true)

# Extract the lineages from simulation results.
saveΔt = 1.0
agents = Dict()
lineages = []
for agent in values(res.final_pop[C])
    lin = Dict()
    lineage = AgentBasedModeling.lineage(res, agent)[1:end-1]
    for a in lineage
        trange = a.btime:saveΔt:a.dtime
        adict = Dict(
            :fluor => a.simulation(trange; idxs = P).u,
            :size => a.simulation(trange; idxs = s).u)
        lin[a.uid] = adict
        agents[a.uid] = adict
    end
    push!(lineages, lin)
end

#Plot results
pt_cm = 2.83465
fig = Figure(size=(170, 50) .* pt_cm; fontsize=8, pt_per_unit=1)
ax_protein = Axis(fig[1,1]; xlabel="Birth protein distribution", ylabel="Probability density")
ax_size = Axis(fig[1,2]; xlabel="Birth size distribution", ylabel="Probability density")

hidedecorations!(ax_protein, ticklabels = false, ticks = false, label = false)
hidespines!(ax_protein, :t, :r) 

hidedecorations!(ax_size, ticklabels = false, ticks = false, label = false)
hidespines!(ax_size, :t, :r) 

sizehist = normalize(fit(Histogram, getfield.(res.prods[:Cs], :value), range(0.2, stop=0.8, length=50)); mode=:pdf)
stairs!(ax_size, collect(midpoints(sizehist.edges[1])), sizehist.weights; color=(colors[1], stairstransp), step=:center, label="Agent-based simulation")
barplot!(ax_size, collect(midpoints(sizehist.edges[1])), sizehist.weights; 
    color=(colors[1], barstransp), strokecolor=(colors[1], transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)

proteinsnap = normalize(fit(Histogram, [Dict(a.init_trait)[P] for a in values(res.final_pop[C])], range(0, stop=100, length=100)); mode=:pdf)
stairs!(ax_protein, collect(midpoints(proteinsnap.edges[1])), proteinsnap.weights; color=(colors[1], stairstransp), step=:center, label="Agent-based simulation")
barplot!(ax_protein, collect(midpoints(proteinsnap.edges[1])), proteinsnap.weights; 
    color=(colors[1], barstransp), strokecolor=(colors[1], transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)

# Comparison with analytical computations.
include("analyticals.jl")
lines!(ax_size, range(span_[1], stop=span_[2], length=N), psi_tree_.(range(span_[1], stop=span_[2], length=N)); color=colors[10], label="Analytical solution")
lines!(ax_protein, xs_, mat_s ./ mat_sx; color=colors[10], label="Analytical solution")

function plot_lineage(axsize, axprotein, agent; kwargs...)
    lin = AgentBasedModeling.lineage(res, agent)
    for l in lin[end-8:end]
        sim = l.simulation
        tstart = l.btime
        if !isnothing(l.dtime)
            tend = l.dtime
            lines!(axsize, sim(tstart:0.1:tend).t, sim(tstart:0.1:tend; idxs=s).u; kwargs...)
            lines!(axprotein, sim(tstart:0.1:tend).t, sim(tstart:0.1:tend; idxs=P).u; kwargs...)
        else
            tend = sim.t[end]
            lines!(axsize, sim(tstart:0.1:tend).t, sim(tstart:0.1:tend; idxs=s).u; kwargs...)
            lines!(axprotein , sim(tstart:0.1:tend).t, sim(tstart:0.1:tend; idxs=P).u; kwargs...)
        end
    end
end

ax_traj_protein = Axis(fig[1,1]; xlabel="Time", ylabel="Protein", title = "Lineage trajectories",
    width=Relative(0.4),
    height=Relative(0.4),
    xlabelsize=6,
    ylabelsize=6,
    xticklabelsize=6,
    yticklabelsize=6,
    xticksize=4,
    yticksize=4,
    titlesize=6,
    halign=1.0,
    valign=1.0,)

ax_traj_size = Axis(fig[1,2]; xlabel="Time", ylabel="Size", title = "Lineage trajectories",
    width=Relative(0.4),
    height=Relative(0.4),
    xlabelsize=6,
    ylabelsize=6,
    xticklabelsize=6,
    yticklabelsize=6,
    xticksize=4,
    yticksize=4,
    titlesize=6,
    halign=1.0,
    valign=1.0,)

i = 1
for agent in rand(last.(collect(Iterators.flatten(values(res.final_pop)))), 50)
    plot_lineage(ax_traj_size, ax_traj_protein, agent; color = (colors[i], htransp), linewidth=0.8)
    global i = mod(i + 1, length(colors)) + 1
end

agent = rand(last.(collect(Iterators.flatten(values(res.final_pop)))))
plot_lineage(ax_traj_size, ax_traj_protein, agent; color = colors[1], linewidth=1.2)
hidedecorations!(ax_traj_size, ticklabels = false, ticks = false, label = false)
hidespines!(ax_traj_size, :t, :r) 

hidedecorations!(ax_traj_protein, ticklabels = false, ticks = false, label = false)
hidespines!(ax_traj_protein, :t, :r) 

xlims!(ax_size, (0.2, 1.0))
xlims!(ax_protein, (0, 100))
xlims!(ax_traj_protein, low=6)
xlims!(ax_traj_size, low=6)
ylims!(ax_traj_protein, high=90)
ylims!(ax_traj_size, high=1.2)

mkpath("$(plotsdir())/size_control/")
save("$(plotsdir())/size_control/sizecontrol_traj.pdf", fig; pt_per_unit = 1.0)
