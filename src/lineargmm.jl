struct IteratedLinearGMM{TF} <: AbstractGMMEstimator{Nothing,TF}
    iter::RefValue{Int}
    Q::RefValue{TF}
    θlast::Vector{TF}
    diff::RefValue{TF}
    Ys::Vector{Vector{TF}}
    Xs::Vector{Matrix{TF}}
    Zs::Vector{Matrix{TF}}
    ZX::Matrix{TF}
    ZY::Vector{TF}
    Winv::Matrix{TF}
    Winvfac::RefValue{Cholesky{TF,Matrix{TF}}}
    resids::Vector{Vector{TF}}
    H::Matrix{TF}
    WZY::Vector{TF}
    XZWZY::Vector{TF}
    WZX::Matrix{TF}
    XZWZX::Matrix{TF}
    G::Vector{TF}
    WG::Vector{TF}
end

function IteratedLinearGMM(eqs, nobs::Integer; TF::Type=Float64)
    nparam = sum(eq->length(eq[2]), eqs)
    nmoment = sum(eq->length(eq[3]), eqs)
    if nmoment == nparam
        throw(ArgumentError("consider JustIdentifiedLinearGMM instead"))
    elseif nmoment < nparam
        throw(ArgumentError("$nmoment moment conditions for $nparam parameters is not allowed (underidentified)"))
    end
    θlast = zeros(TF, nparam)
    Ys = [Vector{TF}(undef, nobs) for _ in eqs]
    Xs = [Matrix{TF}(undef, nobs, length(eq[2])) for eq in eqs]
    Zs = [Matrix{TF}(undef, nobs, length(eq[3])) for eq in eqs]
    ZX = zeros(TF, nmoment, nparam)
    ZY = Vector{TF}(undef, nmoment)
    Winv = Matrix{TF}(undef, nmoment, nmoment)
    Winvfac = RefValue{Cholesky{TF,Matrix{TF}}}()
    resids = [Vector{TF}(undef, nobs) for _ in eqs]
    H = Matrix{TF}(undef, nobs, nmoment)
    WZY = Vector{TF}(undef, nmoment)
    XZWZY = Vector{TF}(undef, nparam)
    WZX = Matrix{TF}(undef, nmoment, nparam)
    XZWZX = Matrix{TF}(undef, nparam, nparam)
    G = Vector{TF}(undef, nmoment)
    WG = Vector{TF}(undef, nmoment)
    return IteratedLinearGMM(Ref(0), Ref(NaN), θlast, Ref(NaN), Ys, Xs, Zs, ZX, ZY,
        Winv, Winvfac, resids, H, WZY, XZWZY, WZX, XZWZX, G, WG)
end

# Assume all equations have the same number of observations
function _filldata!(Ys, Xs, Zs, ZX, ZY, eqs, data)
    for (i, eq) in enumerate(eqs)
        Y, X, Z = Ys[i], Xs[i], Zs[i]
        copyto!(Y, Tables.getcolumn(data, eq[1]))
        for (j, n) in enumerate(eq[2])
            if n == :cons
                fill!(view(X,:,j), 1)
            else
                copyto!(view(X,:,j), Tables.getcolumn(data, n))
            end
        end
        for (j, n) in enumerate(eq[3])
            if n == :cons
                fill!(view(Z,:,j), 1)
            else
                copyto!(view(Z,:,j), Tables.getcolumn(data, n))
            end
        end
        # ZX is already filled with 0
        mul!(view(ZX, 1:length(eq[3]), 1:length(eq[2])), Z', X)
        mul!(view(ZY, 1:length(eq[3])), Z', Y)
    end
    nobs = length(Ys[1])
    ZX ./= nobs
    ZY ./= nobs
end

"""
    fit(::Type{<:IteratedLinearGMM}, vce::CovarianceEstimator, data, eqs; kwargs...)

Conduct linear iterated GMM estimation with weight matrix in each iteration
evaluated as the inverse of variance-covariance matrix estimated by `vce`.
`data` is a `Tables.jl`-compatible data table.
`eqs` specifies the names of the variables.
This method is for the case where the parameters are over-identified.
See documentation website for details.

# Keywords
- `nocons::Bool=false`: do not add constant terms automatically.
- `winitial=:TSLS`: initial weight matrix; use the one for two-stage least squares by default.
- `θtol::Real=1e-8`: tolerance level for determining the convergence of parameters.
- `maxiter::Integer=10000`: maximum number of iterations allowed.
- `showtrace::Bool=false`: print information as iteration proceeds.
- `initonly::Bool=false`: initialize the returned object without conducting the estimation.
- `TF::Type=Float64`: type of the numerical values.
"""
function fit(::Type{<:IteratedLinearGMM}, vce::CovarianceEstimator, data, eqs;
        nocons::Bool=false, winitial=:TSLS, θtol::Real=1e-8, maxiter::Integer=10000,
        showtrace::Bool=false, initonly::Bool=false, TF::Type=Float64)
    Tables.istable(data) ||
        throw(ArgumentError("data must be a Tables.jl-compatible table"))
    nobs = length(Tables.rows(data))
    eqs, params = _parse_eqs(eqs, nocons)
    est = IteratedLinearGMM(eqs, nobs; TF=TF)
    coef = copy(est.θlast)
    nparam = length(coef)
    vcov = Matrix{TF}(undef, nparam, nparam)
    _filldata!(est.Ys, est.Xs, est.Zs, est.ZX, est.ZY, eqs, data)
    if winitial == :TSLS
        fill!(est.Winv, 0)
        r0 = 0
        for Z in est.Zs
            ir = r0+1:r0+size(Z,2)
            mul!(view(est.Winv, ir, ir), Z', Z)
            r0 += size(Z, 2)
        end
        est.Winv ./= nobs # Optional scaling that doesn't matter
        # Winv is assumed to be inverted just for the first iteration
        inv!(cholesky!(est.Winv))
    else
        copyto!(est.Winv, winitial)
    end
    m = LinearGMM(coef, vcov, est, vce, eqs, params)
    initonly || fit!(m; winitial=winitial, θtol=θtol, maxiter=maxiter, showtrace=showtrace)
    return m
end

function setvcov!(m::LinearGMM{<:IteratedLinearGMM}, state)
    est = m.est
    # Preserve the W used for point estimate
    WG = state == 1 ? mul!(m.vce.vcovcache1, est.Winv, est.ZX) :
        ldiv!(m.vce.vcovcache1, est.Winvfac[], est.ZX)
    GWG = mul!(m.vcov, est.ZX', WG)
    # Cannot directly use WG' as adjoint matrix is not allowed with ldiv!
    GW = copyto!(m.vce.vcovcache2, WG')
    GWGGW = ldiv!(cholesky!(Hermitian(GWG)), GW)
    mul!(m.vce.vcovcache1, m.vce.S, GWGGW')
    mul!(m.vcov, GWGGW, m.vce.vcovcache1)
    # ! H is vertical
    m.vcov ./= size(est.H, 1)
end

function setH!(H, resids, Ys, Xs, Zs, coef, eqs)
    r0 = 0
    k0 = 0
    for (i, eq) in enumerate(eqs)
        copyto!(resids[i], Ys[i])
        mul!(resids[i], Xs[i], view(coef, k0+1:k0+length(eq[2])), -1.0, 1.0)
        H = view(H, :, r0+1:r0+length(eq[3]))
        H .= Zs[i] .* resids[i]
        r0 += length(eq[3])
        k0 += length(eq[2])
    end
    return H
end

function iterate(m::LinearGMM{<:IteratedLinearGMM,VCE}, state=1) where VCE
    est = m.est
    if state > 1
        copyto!(est.Winv, m.vce.S)
        est.Winvfac[] = cholesky!(Hermitian(est.Winv))
        copyto!(est.θlast, m.coef)
    end
    state == 1 ? mul!(est.WZY, est.Winv, est.ZY) : ldiv!(est.WZY, est.Winvfac[], est.ZY)
    mul!(est.XZWZY, est.ZX', est.WZY)
    state == 1 ? mul!(est.WZX, est.Winv, est.ZX) : ldiv!(est.WZX, est.Winvfac[], est.ZX)
    mul!(est.XZWZX, est.ZX', est.WZX)
    ldiv!(m.coef, cholesky!(Hermitian(est.XZWZX)), est.XZWZY)
    setH!(est.H, est.resids, est.Ys, est.Xs, est.Zs, m.coef, m.eqs)
    sum!(est.G, est.H')
    est.G ./= size(est.H, 1)
    state == 1 ? mul!(est.WG, est.Winv, est.G) : ldiv!(est.WG, est.Winvfac[], est.G)
    est.Q[] = est.G'est.WG
    setS!(m.vce, est.H')
    est.iter[] += 1
    return m, state+1
end

function fit!(m::LinearGMM{<:IteratedLinearGMM};
        θtol::Real=1e-8, maxiter::Integer=10000, showtrace::Bool=false, kwargs...)
    iter = 0
    while iter < maxiter
        iter += 1
        iterate(m, iter)
        test_θtol!(m.est, m.coef, θtol) && break
        showtrace && _show_trace(stdout, m.est, true, false)
    end
    showtrace && _show_trace(stdout, m.est, true, false)
    try
        setvcov!(m, iter)
    catch
        @warn "variance-covariance matrix is not computed"
    end
end

# ! H is vertical
Jstat(est::IteratedLinearGMM) =
    nmoment(est) > nparam(est) ? size(est.H, 1) * est.Q[] : NaN

show(io::IO, ::MIME"text/plain", est::IteratedLinearGMM; twolines::Bool=false) =
    (println(io, "Iterated Linear GMM estimator:"); print(io, "  ");
        _show_trace(io, est, false, twolines))

struct JustIdentifiedLinearGMM{TF} <: AbstractGMMEstimator{Nothing,TF}
    Ys::Vector{Vector{TF}}
    Xs::Vector{Matrix{TF}}
    Zs::Vector{Matrix{TF}}
    ZX::Matrix{TF}
    ZXinv::Matrix{TF}
    ZY::Vector{TF}
    resids::Vector{Vector{TF}}
    H::Matrix{TF}
end

function JustIdentifiedLinearGMM(eqs, nobs::Integer; TF::Type=Float64)
    nparam = sum(eq->length(eq[2]), eqs)
    nmoment = sum(eq->length(eq[3]), eqs)
    nparam == nmoment || throw(ArgumentError("model is not just identified"))
    Ys = [Vector{TF}(undef, nobs) for _ in eqs]
    Xs = [Matrix{TF}(undef, nobs, length(eq[2])) for eq in eqs]
    Zs = [Matrix{TF}(undef, nobs, length(eq[3])) for eq in eqs]
    ZX = zeros(TF, nmoment, nparam)
    ZXinv = similar(ZX)
    ZY = Vector{TF}(undef, nmoment)
    resids = [Vector{TF}(undef, nobs) for _ in eqs]
    H = Matrix{TF}(undef, nobs, nmoment)
    return JustIdentifiedLinearGMM(Ys, Xs, Zs, ZX, ZXinv, ZY, resids, H)
end

const LinearGMMEstimator{TF} = Union{IteratedLinearGMM{TF}, JustIdentifiedLinearGMM{TF}}

nparam(est::LinearGMMEstimator) = size(est.ZX, 2)
nmoment(est::LinearGMMEstimator) = size(est.ZX, 1)

"""
    fit(::Type{<:JustIdentifiedLinearGMM}, vce::CovarianceEstimator, data, eqs; kwargs...)

Conduct just-identified linear GMM estimation
with variance-covariance matrix estimated by `vce`.
`data` is a `Tables.jl`-compatible data table.
`eqs` specifies the names of the variables.
See documentation website for details.

# Keywords
- `nocons::Bool=false`: do not add constant terms automatically.
- `initonly::Bool=false`: initialize the returned object without conducting the estimation.
- `TF::Type=Float64`: type of the numerical values.
"""
function fit(::Type{<:JustIdentifiedLinearGMM}, vce::CovarianceEstimator, data, eqs;
        nocons::Bool=false, initonly::Bool=false, TF::Type=Float64)
    Tables.istable(data) ||
        throw(ArgumentError("data must be a Tables.jl-compatible table"))
    nobs = length(Tables.rows(data))
    eqs, params = _parse_eqs(eqs, nocons)
    est = JustIdentifiedLinearGMM(eqs, nobs; TF=TF)
    coef = similar(est.ZY)
    nparam = length(coef)
    vcov = Matrix{TF}(undef, nparam, nparam)
    _filldata!(est.Ys, est.Xs, est.Zs, est.ZX, est.ZY, eqs, data)
    m = LinearGMM(coef, vcov, est, vce, eqs, params)
    initonly || fit!(m)
    return m
end

function setvcov!(m::LinearGMM{<:JustIdentifiedLinearGMM})
    est = m.est
    mul!(m.vce.vcovcache1, m.vce.S, est.ZXinv')
    mul!(m.vcov, est.ZXinv, m.vce.vcovcache1)
    # ! H is vertical
    m.vcov ./= size(est.H, 1)
end

function fit!(m::LinearGMM{<:JustIdentifiedLinearGMM})
    est = m.est
    # ZXinv will be reused for vcov
    copyto!(est.ZXinv, est.ZX)
    mul!(m.coef, inv!(lu!(est.ZXinv)), est.ZY)
    setH!(est.H, est.resids, est.Ys, est.Xs, est.Zs, m.coef, m.eqs)
    setS!(m.vce, est.H')
    try
        setvcov!(m)
    catch
        @warn "variance-covariance matrix is not computed"
    end
end

show(io::IO, ::MIME"text/plain", est::JustIdentifiedLinearGMM; kwargs...) =
    print(io, "Just-identified linear GMM estimator")
