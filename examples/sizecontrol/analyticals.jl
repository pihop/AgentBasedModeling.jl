using QuadGK
using ModelingToolkit
using Distributions
using CairoMakie
using LinearAlgebra
using Integrals
using Intervals
using Interpolations

α_ = 0.05
kprod_ = 2.0 
b_ = 2.0
μ_ = 1.0
cv_ = 0.2 

lowTriIter2(A::AbstractMatrix) = 
  Iterators.flatten(Iterators.map(j->(j,i), i:size(A,1)) for i=1:size(A,2))

intparams = ( 
    abstol = 1e-4,
    reltol = 1e-4)

γ(s, s0) = 1/(α_*s) * γdiv(μ_, cv_, s, s0)
γint(s, s0) = IntegralProblem((u,p) -> γ(u, s0), (s0, s))
phi(s, s0) = γ(s, s0) * exp(-solve(γint(s, s0), QuadGKJL()).u)

ddist = Beta(7, 7)

kernelf(s, s0) = IntegralProblem((u,p) -> 2*pdf(ddist, u) * phi(s/u, s0), (0.0001, 1.0))
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
    x = range(a, stop=b, step=h)
    
    Xi = Float64[]
    Ai = zeros(length(x)-1, length(x)-1)
    
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

nbins=100

span_ = (0.0001, 0.6)
N = 50
sstep_ = (span_[2]-span_[1])/N
srange_ = collect(range(span_[1], stop=span_[2], length=N))
A = volterra(ker, span_, N)
sn = trapz(Float64.(eigvecs(A)[:,end]), sstep_)

psiA_tree = Float64.(eigvecs(A)[:,end]) ./ srange_
sn_tree = trapz(psiA_tree, sstep_)

psi = linear_interpolation(
    range(span_[1], stop=span_[2], length=N), Float64.(eigvecs(A)[:,end]) .* 1/sn)
psi_tree_ = linear_interpolation(
    range(span_[1], stop=span_[2], length=N), psiA_tree .* 1/sn_tree)
# Compute protein distribution for different birth sizes s0.

function rho(s_, s0_, s0)
    (1/psi(s0))*pdf(ddist, s0/s_)*phi(s_, s0_)*psi(s0_)
end

# Birth protein counts... Use concentration homeostasis. 
dist(kprod, b, α, s) = NegativeBinomial(kprod/(α*s), 1/(2+b*s)) 
Pi(x, s) = pdf(dist(kprod_, b_, α_, s), x)
B(x, x_, θ) = pdf(Binomial(x_, θ), x)

# Integrate
iprob_s0(x, x_, s_, s0) = IntegralProblem((u, p) -> B(x, x_, s0/s_)*rho(s_, u, s0)*Pi(x_, s_), span_)

idist(x, x_, s0) = solve(
    IntegralProblem((u, p) -> solve(iprob_s0(x, x_, u, s0), QuadGKJL(); intparams...).u, (s0, span_[end])), QuadGKJL(); intparams...).u

xs_ = floor.(collect(range(0, stop=200, length=20)))
xstep_ = xs_[2] - xs_[1]
array_dists(x, s0) = trapz(map(x_ -> idist(x, x_, s0), xs_), xstep_)
ss_ = range(span_[1], stop=span_[2], length=20)

mat = [array_dists(x, s)*psi_tree_(s) for x in xs_, s in ss_]
mat_s = trapz.(eachrow(mat), Ref(sstep_))
mat_sx = trapz(mat_s, xstep_)
