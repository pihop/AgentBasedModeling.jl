using QuadGK
using ModelingToolkit
using Distributions
using CairoMakie
using LinearAlgebra
using Integrals
using Intervals
using Interpolations

α_ = 1.0
kprod_ = 10.0 
b_ = 6.0
μ_ = 1.0
cv_ = 0.2 

span_ = (1e-4, 0.8)

intparams = ( 
    abstol = 1e-4,
    reltol = 1e-4)

γ(s, s0) = γdiv(μ_, cv_, s, s0)
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
        print("$i \r")
    end
    return Ai
end

N = 50
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

sintspan_ = [span_[1], 1.8]
# Birth protein counts... Use concentration homeostasis. 
dist(kprod, b, α, s) = NegativeBinomial(kprod/(α), 1/(1+b*s)) 
Pi(x, s) = pdf(dist(kprod_, b_, α_, s), x)
B(x, x_, θ) = pdf(Binomial(x_, θ), x)

# Integrate
rhoint(x, x_, s_, s0) = IntegralProblem((u, p) ->  rho(s_, u, s0), sintspan_)
Pi0int(x, x_, s0) = IntegralProblem(
    (u,p) -> B(x, x_, s0/u)*solve(rhoint(x, x_, u, s0), QuadGKJL(); intparams...).u *Pi(x_, u), (s0, sintspan_[end]))
Pi0(x, x_, s0)  = solve(Pi0int(x, x_, s0), QuadGKJL(); intparams...).u

xs_ = collect(range(0, stop=150, step=5))
xstep_ = xs_[2] - xs_[1]
array_dists(x, s0) = map(x_ -> Pi0(x, x_, s0), xs_)
ss_ = range(sintspan_[1], stop=sintspan_[2], length=10)

mat = [array_dists(x, s) for x in xs_, s in ss_]
mat_ = hcat([row .* psi_tree_.(ss_) for row in eachrow(mat)]...)'
mat_ = trapz.(mat_, Ref(xstep_))
mat_s = trapz.(eachrow(mat_), Ref(sstep_))
mat_sx = trapz(mat_s, xstep_)
