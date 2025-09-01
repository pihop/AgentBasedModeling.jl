using CairoMakie
using JLD2
using ColorSchemes
using StatsBase
using Distributions
colors = ColorSchemes.Hiroshige.colors
transp = 0.2
stairstransp = 0.9

using Catalyst
using JumpProcesses
using LinearAlgebra

res = load("data/sir/sir_results.jld2", "ares");

timesa = []
for r in res.u
    idx_ = findfirst(x -> x[1] == 0, r.u)
    isnothing(idx_) && continue
    push!(timesa, r.t[idx_]) 
end

R0_ = 1.5
ε_ = 0.04
μ_ = 0.02
γ_ = μ_/ε_ - μ_
β_ = R0_ / (γ_ + μ_)
scale = 1 
N = Int64(500)
I0 = 1 
T = 100.
tspan = (0, T)
Δt = 1 

K_ = 0.15
n_ = 3.0

standard_sir = @reaction_network begin
    @parameters μ γ β
    @species S(t) I(t) R(t)
    μ*(S+I+R), 0 --> S
    μ, S --> 0
    μ, I --> 0
    μ, R --> 0
    β/(S+I+R), S + I --> 2 * I
    γ, I --> R
end
p  = (:μ => μ_, :γ => γ_, :β => β_)
u0 = [:I => I0, :S => N - I0, :R => 0]
dprob = DiscreteProblem(standard_sir, u0, tspan, p)
jprob = JumpProblem(standard_sir, dprob, Direct())
eprob = EnsembleProblem(jprob)
esol = solve(eprob, SSAStepper(), EnsembleThreads(); trajectories = 10000)
esum = EnsembleSummary(esol, tspan[1]:Δt:tspan[2]; )

timescme = [] 
for r in esol
    idx_ = findfirst(x -> x[2] == 0, r.u)
    isnothing(idx_) && continue
    push!(timescme, r.t[idx_]) 
end


pt_cm = 2.83465
# 1 cm = 28,3465 pt
# Total size 170x225mm
fig = Figure(size=(170, 50) .* pt_cm; fontsize=8, pt_per_unit=1, figure_padding = 0.1)

axt = Axis(fig[1,2]; xlabel="Time to extinction", ylabel="Proabability density", title="Burnout time")
axtraj = Axis(fig[1,1]; 
    xlabel = "Time", 
    ylabel = "Infected population size"
    )

tcme = normalize(fit(Histogram, timescme, 1:2:100); mode=:pdf)
ta = normalize(fit(Histogram, timesa, 1:2:100); mode=:pdf)

stairs!(axt, collect(midpoints(tcme.edges[1])), tcme.weights; color=(colors[1], stairstransp), step=:center, label="Gillespie")
barplot!(axt, collect(midpoints(tcme.edges[1])), tcme.weights; 
    color=(colors[1], transp), strokecolor=(colors[1], transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)

stairs!(axt, collect(midpoints(ta.edges[1])), ta.weights; color=(colors[10], stairstransp), step=:center, label="Agent-based model")
barplot!(axt, collect(midpoints(ta.edges[1])), ta.weights; 
    color=(colors[10], transp), strokecolor=(colors[10], transp), strokewidth=0.0, gap=0.0, dodge_gap=0.0)

hidedecorations!(axt, ticks=false, label=false, ticklabels=false)
hidespines!(axt, :r, :t)
axislegend(axt; orientation=:vertical, framevisible=false, tellwidth=false, tellheight=false, rowgap = -10, labelsize=6)

xlims!(axt, (1, 100))
ylims!(axt, (0, 0.1))

inset_ax = Axis(fig[1, 1],
    xlabel = "Lookahead",
    ylabel = "Simulation time \n (s/trajctory)",
    width=Relative(0.4),
    height=Relative(0.4),
    halign=1.0,
    valign=1.0)
hidedecorations!(inset_ax, ticks=false, label=false, ticklabels=false)
hidespines!(inset_ax, :r, :t)
ylims!(inset_ax, low=0)

for (i, idx) in zip([1,1], [2,2])
    traj = res.u[idx]
    lines!(axtraj, traj.t, getindex.(traj.u, 1); color=colors[i], linewidth=2.0)
    idx_ = findfirst(x -> x == 0, getindex.(traj.u, 1))
    scatter!(axtraj, traj.t[idx_], 0; color=colors[i], markersize=10, strokecolor=colors[i], strokewidth=0.5)
end

#hidedecorations!(axtraj, ticks=false, label=false, ticklabels=false)
#hidespines!(axtraj, :r, :t)
#
##lines!(inset_ax, getindex.(timesLext , 3), getindex.(timesLext , 1); color=colors[1])
##lines!(inset_ax, getindex.(timesLfrm , 3), getindex.(timesLfrm , 1); color=colors[2])
#
#hidedecorations!(inset_ax, ticks=false, label=false, ticklabels=false)
#hidespines!(inset_ax, :r, :t)
#xlims!(inset_ax, (0, 2))
#ylims!(inset_ax, (10, 50))
#
mkpath("plots/sir/")
save("plots/sir/burn.pdf", fig, pt_per_unit=1)
