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
using SciMLBase

restraj = load("$(datadir())/sir/trajectories.jld2", "trajectories");
res = load("$(datadir())/sir/sir_results.jld2", "ares");

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
N = Int64(1000)
I0 = 3 
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

stairs!(axt, collect(midpoints(tcme.edges[1])), ta.weights; color=(colors[10], stairstransp), step=:center, label="Agent-based model")
barplot!(axt, collect(midpoints(tcme.edges[1])), ta.weights; 
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

getts(traj, sym) = [x.time for x in traj.snapshot[sym]]
getval(traj, sym) = [x.values for x in traj.snapshot[sym]]

for (i,idx) in zip([2,8], [2,5])
    lines!(axtraj, getts(restraj[idx], :I), getval(restraj[idx], :I); color=colors[i], linewidth=2.0)
    idx_ = findfirst(x -> x == 0, getval(restraj[idx], :I))
    scatter!(axtraj, getts(restraj[idx], :I)[idx_], 0; color=colors[i], markersize=10, marker=:circ, strokecolor=colors[i], strokewidth=0.5)
end

hidedecorations!(axtraj, ticks=false, label=false, ticklabels=false)
hidespines!(axtraj, :r, :t)

#lines!(inset_ax, getindex.(timesLext , 3), getindex.(timesLext , 1); color=colors[1])
#lines!(inset_ax, getindex.(timesLfrm , 3), getindex.(timesLfrm , 1); color=colors[2])

hidedecorations!(inset_ax, ticks=false, label=false, ticklabels=false)
hidespines!(inset_ax, :r, :t)
xlims!(inset_ax, (0, 2))
ylims!(inset_ax, (10, 50))

mkpath("$(plotsdir())/sir/")
save("$(plotsdir())/sir/burn.pdf", fig, pt_per_unit=1)
