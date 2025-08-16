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

function trapz(f::Function, xs)
    fx = f.(xs) 
    fxx = zip(fx, fx[2:end]) 
    fxx = [(a + b)/2 for (a, b) in fxx]
    xss = diff(xs) 
    out = 0.0
    for (f_, x_) in zip(fxx, xss)
        out += f_ * x_ 
    end
    return out
end

function trapz(fx::Vector, xs)
    out = 0.0
    fxx = zip(fx, fx[2:end])
    fxx = [(a + b)/2 for (a, b) in fxx]
    xss = diff(xs) 
    for (f_, x_) in zip(fxx, xss)
        out += f_ * x_ 
    end
    return out
end

ss_ = range(span_[1], stop=span_[2], length=20)

γ(s, s0) = γdiv(μ_, cv_, s, s0)
γint(s, s0) = IntegralProblem((u,p) -> γ(u, s0), span_)
phi(s, s0) = γ(s, s0) * exp(-trapz(u -> γ(u, s0), ss_))

ddist = Beta(100)

kernelf(s, s0) = IntegralProblem((u,p) -> 2*pdf(ddist, u) * phi(s/u, s0), span_)
ker(s, s0) = trapz(u -> 2*pdf(ddist, u)*phi(s/u, s0), ss_)

function volterra(ker, ss)
    h = diff(ss)
    x = ss
    n = length(ss) - 1
    
    Xi = Float64[]
    Ai = zeros(n, n)
    
    for i in 1:n
        for j in range(1, n, step=1)
           Ai[i,j] = h[i]*ker(x[i], x[j])
        end
        Ai[1,i] = h[i]*ker(x[1],x[i])
        Ai[i,i] = h[i]*ker(x[i],x[i])
        print("$i \r")
    end
    return Ai
end


