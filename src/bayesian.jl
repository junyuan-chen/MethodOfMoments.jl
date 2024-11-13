struct BayesianGMM{PR<:Tuple, DF, VCE, G, DG, PG, PDG, P,
        TF<:AbstractFloat} <: AbstractGMMEstimator{P,TF}
    coef::Vector{TF}
    vce::VCE
    l::RefValue{TF}
    dl::Vector{TF}
    g::G
    dg::DG
    preg::PG
    predg::PDG
    H::Matrix{TF}
    G::Vector{TF}
    WG::Vector{TF}
    dG::Matrix{TF}
    W::Matrix{TF}
    p::P
    params::Vector{VarName}
    priors::PR
    deriv::DF
    dprior::Vector{TF}
end

_parse_deriv(deriv::Nothing) = deriv
_parse_deriv(deriv::Symbol) = Val(deriv)
_parse_deriv(deriv::Module) = Val(nameof(deriv))

"""
    BayesianGMM(vce::CovarianceEstimator,
        g, dg, params, nmoment::Integer, nobs::Integer; kwargs...)

Construct an object that supports the Bayesian quasi-likelihood evaluations.
Arguments provided are defined in the same way as for nonlinear iterated GMM estimation.
See documentation website for details.

# Keywords
- `preg=nothing`: a function for processing the data frame before evaluating moment conditions.
- `predg=nothing`: a function for processing the data frame before evaluating the derivatives for moment conditions.
- `deriv=nothing`: specify how gradients for the quasi-posterior functions are computed.
- `ntasks::Integer=_default_ntasks(nobs*nmoment)`: number of threads use for evaluating moment conditions and their derivatives across observations.
- `TF::Type=Float64`: type of the numerical values.
"""
function BayesianGMM(vce::CovarianceEstimator, g, dg,
        params, nmoment::Integer, nobs::Integer;
        preg=nothing, predg=nothing, deriv=nothing,
        ntasks::Integer=_default_ntasks(nobs*nmoment), TF::Type=Float64)
    nparam = length(params)
    coef = Vector{TF}(undef, nparam)
    dl = Vector{TF}(undef, nparam)
    # H is horizontal
    H = Matrix{TF}(undef, nmoment, nobs)
    G = Vector{TF}(undef, nmoment)
    WG = Vector{TF}(undef, nmoment)
    dG = Matrix{TF}(undef, nmoment, nparam)
    W = Matrix{TF}(undef, nmoment, nmoment)
    ntasks = min(ntasks, nobs)
    if ntasks > 1
        step = nobs ÷ ntasks
        rowcuts = Int[(1:step:1+step*(ntasks-1))..., nobs+1]
        Gs = [Vector{TF}(undef, nmoment) for _ in 1:ntasks]
        dGs = [Matrix{TF}(undef, nmoment, nparam) for _ in 1:ntasks]
        p = PartitionedGMMTasks(rowcuts, Gs, dGs)
    else
        p = nothing
    end
    params, priors = _parse_bayes_params(params)
    dprior = Vector{TF}(undef, nparam)
    deriv = _parse_deriv(deriv)
    return BayesianGMM(coef, vce, Ref(NaN), dl, g, dg, preg, predg, H, G, WG, dG, W,
        p, params, priors, deriv, dprior)
end

coef(m::BayesianGMM) = m.coef
coef(m::BayesianGMM, n::VarName) = m.coef[findfirst(==(n), m.params)]
coefnames(m::BayesianGMM) = m.params

nparam(m::BayesianGMM) = length(coef(m))
nmoment(m::BayesianGMM) = length(m.G)

const TransformedBayesianGMM{T,L} =
    TransformedLogDensity{T,L} where {T<:AbstractTransform, L<:BayesianGMM}
const BayesOrTrans = Union{BayesianGMM, TransformedBayesianGMM}

transform(transformation, m::BayesianGMM) = TransformedLogDensity(transformation, m)

parent(m::BayesianGMM) = m
parent(m::TransformedBayesianGMM) = m.log_density_function

@inline getindex(m::BayesOrTrans) = parent(m).coef
@inline getindex(m::BayesOrTrans, i) = getindex(m[], i)

function logprior(m::BayesianGMM{PR}, θ=m.coef) where PR
    # The generated part avoids allocations for array θ
    if @generated
        ptypes = PR.parameters
        ex = :(lp = logpdf(m.priors[1], θ[1]))
        for i in 2:length(ptypes)
            ex = :($ex; lp += logpdf(m.priors[$i], θ[$i]))
        end
        return :($ex; lp)
    else
        ptypes = PR.parameters
        lp = logpdf(m.priors[1], θ[1])
        for i in 2:length(ptypes)
            lp += logpdf(m.priors[i], θ[i])
        end
        return lp
    end
end

logprior(m::TransformedBayesianGMM, θ) =
    logprior(parent(m), transform(m.transformation, θ))

struct LogPdf{D}
    d::D
end

(p::LogPdf{D})(θ) where {D<:Distribution} = logpdf(p.d, θ)

struct LogPrior{D,DF}
    d::D
    df::DF
end

logprior_and_gradient!(m::BayesianGMM{PR,Nothing}, θ, grad) where PR =
    error("function for computing derivatives for priors is not specified")

# Fallback method that requires a method of LogPrior for DF
function logprior_and_gradient!(m::BayesianGMM{PR,DF}, θ, grad) where {PR,DF}
    # The generated part avoids allocations for array θ
    if @generated
        ex = :(lp = 0.0)
        for i in 1:length(PR.parameters)
            ex = :($ex; lpi, dlpi = LogPrior(m.priors[$i], m.deriv)(θ[$i]);
                lp += lpi; grad[$i] = dlpi)
        end
        return :($ex; lp, grad)
    else
        lp = 0.0
        for i in 1:length(PR.parameters)
            lpi, dlpi = LogPrior(m.priors[i], m.deriv)(θ[i])
            lp += lpi
            grad[i] = dlpi
        end
        return lp, grad
    end
end

function loglikelihood!(m::BayesianGMM, θ)
    copyto!(m.coef, θ)
    m.preg === nothing || m.preg(θ)
    setG!(m, m.g, θ)
    setS!(m.vce, m.H)
    copyto!(m.W, m.vce.S)
    try
        ldiv!(m.WG, cholesky!(Hermitian(m.W)), m.G)
    catch
        m.l[] = l = -Inf
        return l
    end
    m.l[] = l = -(m.G'm.WG)/2
    return l
end

logposterior!(m::BayesianGMM, θ) = loglikelihood!(m, θ) + logprior(m)

# Does not add the log Jacobian determinant
logposterior!(m::TransformedBayesianGMM, θ) =
    logposterior!(parent(m), transform(m.transformation, θ))

# Needed for TransformedLogDensity
(m::BayesOrTrans)(θ) = logposterior!(m, θ)

# ! TO DO: Handle TransformedBayesianGMM ?
function logposterior_and_gradient!(m::BayesianGMM, θ, grad::AbstractVector)
    l = loglikelihood!(m, θ)
    pr, grad = logprior_and_gradient!(m, θ, grad)
    p = parent(m)
    p.predg === nothing || p.predg(θ)
    setdG!(p, p.dg, θ)
    mul!(grad, p.dG', p.WG, 1/l, 1.0)
    return l + pr, grad
end

logposterior_and_gradient!(m::BayesianGMM, θ) =
    logposterior_and_gradient!(m, θ, m.dl)

function (m::BayesOrTrans)(θ, grad::AbstractVector)
    if length(grad) > 0
        l, grad = logposterior_and_gradient!(m, θ, grad)
        return l
    else
        return logposterior!(m, θ)
    end
end

capabilities(::Type{<:BayesOrTrans}) = LogDensityOrder{1}()
dimension(m::BayesianGMM) = length(m.coef)
logdensity(m::BayesianGMM, θ) = logposterior!(m, θ)

function logdensity_and_gradient(m::BayesianGMM, θ)
    l, dl = logposterior_and_gradient!(m, θ)
    return l, copy(dl)
end

show(io::IO, m::BayesianGMM) =
    print(io, nmoment(m), '×', nparam(m), ' ', typeof(m).name.name)

function show(io::IO, mime::MIME"text/plain", m::BayesianGMM)
    nm = nmoment(m)
    np = nparam(m)
    print(io, typeof(m).name.name, " with ", nm, " moment")
    nm > 1 && print(io, 's')
    print(io, " and ", np, " parameter")
    println(io, np > 1 ? "s:" : ":")
    cf = coef(m)
    coefs = (string(m.params[i],
        @sprintf(" = %.5e", cf[i])) for i in 1:min(length(cf), 8))
    print(io, "  ")
    join(io, coefs, "  ")
    length(cf) > 8 && print(io, " …")
    @printf(io, "\n  log(posterior) = %.5e", m.l[])
end
