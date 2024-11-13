abstract type AbstractGMMEstimator{P, TF<:AbstractFloat} end

abstract type AbstractGMMResult{TF<:AbstractFloat} <: StatisticalModel end

coef(m::AbstractGMMResult) = m.coef
coef(m::AbstractGMMResult, n::VarName) = m.coef[findfirst(==(n), m.params)]
coefnames(m::AbstractGMMResult) = m.params
vcov(m::AbstractGMMResult) = m.vcov
stderror(m::AbstractGMMResult, n::VarName) =
    (i = findfirst(==(n), m.params); sqrt(m.vcov[i,i]))

function confint(m::AbstractGMMResult; level::Real=0.95)
    scale = norminvcdf(1-(1-level)/2)
    se = stderror(m)
    b = coef(m)
    return b .- scale .* se, b .+ scale .* se
end

"""
    nparam(m::AbstractGMMResult)

Return the number of parameters.
"""
nparam(m::AbstractGMMResult) = length(coef(m))

"""
    nmoment(m::AbstractGMMResult)

Return the number of moment conditions.
"""
nmoment(m::AbstractGMMResult) = nmoment(m.est)

"""
    Jstat(m::AbstractGMMResult)

Return the Hansen's J statistic for over-identified GMM.
`NaN` is returned if the parameters are just-identified.
"""
Jstat(m::AbstractGMMResult) = Jstat(m.est)
Jstat(::AbstractGMMEstimator) = NaN

struct VectorObjValue{TE,G,P}
    est::TE
    g::G
    pre::P
end

struct VectorObjJacobian{TE,DG,P}
    est::TE
    dg::DG
    pre::P
end

struct ObjValue{TE,G,P}
    est::TE
    g::G
    pre::P
end

struct ObjGradient{TE,G,P}
    est::TE
    dg::G
    pre::P
end

function coeftable(m::AbstractGMMResult; level::Real=0.95)
    cf = coef(m)
    se = stderror(m)
    ts = cf ./ se
    pv = 2 .* normccdf.(abs.(ts))
    cil, ciu = confint(m, level=level)
    levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
    cnames = coefnames(m)
    return CoefTable(Vector[cf, se, ts, pv, cil, ciu],
        ["Estimate", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        [string(cnames[i]) for i = 1:length(cf)], 4, 3)
end

show(io::IO, m::AbstractGMMResult) =
    print(io, nmoment(m), '×', nparam(m), ' ', typeof(m).name.name)

function show(io::IO, mime::MIME"text/plain", m::AbstractGMMResult)
    nm = nmoment(m)
    np = nparam(m)
    print(io, typeof(m).name.name, " with ", nm, " moment")
    nm > 1 && print(io, 's')
    print(io, " and ", np, " parameter")
    println(io, np > 1 ? "s:" : ":")
    print(io, "  ")
    show(io, mime, m.est; twolines=true)
    println(io, "\n  ", m.vce)
    show(io, coeftable(m))
end

show(io::IO, est::AbstractGMMEstimator) = print(io, typeof(est).name.name)

struct NonlinearGMM{TE<:AbstractGMMEstimator, VCE<:CovarianceEstimator, SOL,
        G, DG, PG, PDG, TF} <: AbstractGMMResult{TF}
    coef::Vector{TF}
    vcov::Matrix{TF}
    g::G
    dg::DG
    preg::PG
    predg::PDG
    est::TE
    vce::VCE
    solver::SOL
    params::Vector{VarName}
end

struct LinearGMM{TE<:AbstractGMMEstimator, VCE<:CovarianceEstimator,
        TF} <: AbstractGMMResult{TF}
    coef::Vector{TF}
    vcov::Matrix{TF}
    est::TE
    vce::VCE
    eqs::Vector{Tuple{VarName,Vector{VarName},Vector{VarName}}} # Y, X, Z for each equation
    params::Vector{VarName}
end
