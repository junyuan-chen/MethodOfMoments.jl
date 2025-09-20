struct CUGMM{P,TF,VCE<:CovarianceEstimator} <: AbstractGMMEstimator{P,TF}
    Q::RefValue{TF}
    H::Matrix{TF}
    G::Vector{TF}
    WG::Vector{TF}
    dG::Matrix{TF}
    W::Matrix{TF}
    Wfac::RefValue{Cholesky{TF,Matrix{TF}}}
    Wup::Matrix{TF}
    p::P
    vce::VCE
end

function CUGMM(nparam::Integer, nmoment::Integer, nobs::Integer, ntasks::Integer,
        vce::CovarianceEstimator; TF::Type=Float64)
    # H is horizontal
    H = Matrix{TF}(undef, nmoment, nobs)
    G = Vector{TF}(undef, nmoment)
    WG = Vector{TF}(undef, nmoment)
    dG = Matrix{TF}(undef, nmoment, nparam)
    W = Matrix{TF}(undef, nmoment, nmoment)
    Wfac = RefValue{Cholesky{TF,Matrix{TF}}}()
    Wfac[] = cholesky!(diagm(0=>ones(TF, nmoment)))
    Wup = similar(W)
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
    return CUGMM(Ref(NaN), H, G, WG, dG, W, Wfac, Wup, p, vce)
end

# ! H is horizontal
nobs(est::CUGMM) = size(est.H, 2)
nparam(est::CUGMM) = size(est.dG, 2)
nmoment(est::CUGMM) = length(est.G)

function _initsolver(::Type{<:Hybrid}, est::CUGMM, g, dg, preg, predg, θ0;
        warn=false, kwargs...)
    nmoment = length(est.G)
    f = VectorObjValue(est, g, preg)
    # Call preg once in case uninitiated values can create NaNs in jacobian
    preg === nothing || preg(θ0)
    # Set thres_jac=0 to recompute the Jacobian in every iteration
    # This seems to be important for the reliability of the results
    return init(Hybrid{LeastSquares}, f, θ0, nmoment;
        warn=warn, thres_jac=0, kwargs...)
end

"""
    fit(::Type{<:CUGMM}, solvertype, vce::CovarianceEstimator,
        g, dg, params, nmoment::Integer, nobs::Integer; kwargs...)

Conduct nonlinear continuous-updating GMM estimation
with a solver of `solvertype` and a variance-covariance matrix estimated by `vce`.
Moment conditions and their derivatives are specified as `g` and `dg`.
Names of the parameters, number of moment conditions
and number of observations are provided as `params`, `nmoment` and `nobs`.
See documentation website for details.

# Keywords
- `preg=nothing`: a function for processing the data frame before evaluating moment conditions.
- `predg=nothing`: a function for processing the data frame before evaluating the derivatives for moment conditions.
- `ntasks::Integer=_default_ntasks(nobs*nmoment)`: number of threads use for evaluating moment conditions and their derivatives across observations.
- `initonly::Bool=false`: initialize the returned object without conducting the estimation.
- `solverkwargs=NamedTuple()`: keyword arguments passed to the optimization solver as a `NamedTuple`.
- `TF::Type=Float64`: type of the numerical values.
"""
function fit(::Type{<:CUGMM}, solvertype, vce::CovarianceEstimator,
        g, dg, params, nmoment::Integer, nobs::Integer;
        preg=nothing, predg=nothing,
        ntasks::Integer=_default_ntasks(nobs*nmoment),
        initonly::Bool=false, solverkwargs=NamedTuple(), TF::Type=Float64)
    checksolvertype(solvertype)
    params, θ0 = _parse_params(params)
    nparam = length(params)
    dg = _initdg(dg, g, params, nmoment)
    est = CUGMM(nparam, nmoment, nobs, ntasks, vce; TF=TF)
    # solver obj and jac are handled within _initsolver
    solver = _initsolver(solvertype, est, g, dg, preg, predg, θ0; solverkwargs...)
    coef = copy(θ0)
    vcov = Matrix{TF}(undef, nparam, nparam)
    m = NonlinearGMM(coef, vcov, g, dg, preg, predg, est, vce, solver, params)
    initonly || fit!(m)
    return m
end

function (f::VectorObjValue{<:CUGMM})(F, θ)
    est = f.est
    f.pre === nothing || f.pre(θ)
    setG!(est, f.g, θ)
    setS!(est.vce, est.H)
    copyto!(est.W, est.vce.S)
    inv!(cholesky!(est.W))
    W1 = est.Wfac[].factors
    copyto!(W1, est.W)
    est.Wfac[] = Wch = cholesky!(Hermitian(W1))
    copyto!(est.Wup, Wch.UL)
    mul!(F, est.Wup, est.G)
end

# Required by the NLopt ext
function (f::ObjValue{<:CUGMM})(θ)
    est = f.est
    f.pre === nothing || f.pre(θ)
    setG!(est, f.g, θ)
    setS!(est.vce, est.H)
    copyto!(est.W, est.vce.S)
    inv!(cholesky!(est.W))
    mul!(est.WG, est.W, est.G)
    est.G'est.WG
end

function setvcov!(m::NonlinearGMM{<:CUGMM})
    est = m.est
    nmoment, nparam = size(est.dG)
    if nmoment == nparam
        copyto!(m.vce.vcovcache1, est.dG)
        dGinv = inv!(lu!(m.vce.vcovcache1))
        mul!(m.vce.vcovcache2, m.vce.S, dGinv')
        mul!(m.vcov, dGinv, m.vce.vcovcache2)
    else
        # Preserve the W used for point estimate
        WG = mul!(m.vce.vcovcache1, est.W, est.dG)
        GWG = mul!(m.vcov, est.dG', WG)
        # Cannot directly use WG' as adjoint matrix is not allowed with ldiv!
        GW = copyto!(m.vce.vcovcache2, WG')
        GWGGW = ldiv!(cholesky!(Hermitian(GWG)), GW)
        mul!(m.vce.vcovcache1, m.vce.S, GWGGW')
        mul!(m.vcov, GWGGW, m.vce.vcovcache1)
    end
    m.vcov ./= nobs(m)
end

"""
    fit!(m::NonlinearGMM{<:CUGMM}; kwargs...)
    fit!(m::LinearGMM{<:LinearCUGMM}; kwargs...)

An in-place version of [`fit`](@ref) with preallocated `m`.
"""
function fit!(m::NonlinearGMM{<:CUGMM,VCE,<:NonlinearSystem}; kwargs...) where VCE
    solve!(m.solver)
    copyto!(m.coef, m.solver.x)
    # Last evaluation may not be at coef if the trial is rejected
    m.preg === nothing || m.preg(m.coef)
    est = m.est
    setG!(est, m.g, m.coef)
    setS!(est.vce, est.H)
    copyto!(est.W, est.vce.S)
    inv!(cholesky!(est.W))
    # Solver does not update dG in every step
    m.predg === nothing || m.predg(m.coef)
    setdG!(est, m.dg, m.coef)
    mul!(est.WG, est.W, est.G)
    est.Q[] = est.G'est.WG
    try
        setvcov!(m)
    catch
        @warn "variance-covariance matrix is not computed"
    end
    return m
end

struct LinearCUGMM{TF,VCE} <: AbstractGMMEstimator{Nothing,TF}
    eqs::Vector{Tuple{VarName,Vector{VarName},Vector{VarName}}}
    Q::RefValue{TF}
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
    Wup::Matrix{TF}
    vce::VCE # Need vce here for VectorObjValue and ObjValue
end

# ! H is vertical
nobs(est::LinearCUGMM) = size(est.H, 1)
nparam(est::LinearCUGMM) = size(est.ZX, 2)
nmoment(est::LinearCUGMM) = size(est.ZX, 1)

function _initsolver(::Type{<:Hybrid}, est::LinearCUGMM, g, dg, preg, predg, θ0;
        warn=false, kwargs...)
    nmoment = length(est.G)
    f = VectorObjValue(est, nothing, nothing)
    return init(Hybrid{LeastSquares}, f, θ0, nmoment; warn=warn, kwargs...)
end

function LinearCUGMM(eqs, nobs::Integer, vce; TF::Type=Float64)
    nparam = sum(eq->length(eq[2]), eqs)
    nmoment = sum(eq->length(eq[3]), eqs)
    if nmoment < nparam
        throw(ArgumentError("$nmoment moment conditions for $nparam parameters is not allowed (underidentified)"))
    end
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
    Wup = Matrix{TF}(undef, nmoment, nmoment)
    return LinearCUGMM(eqs, Ref(NaN), Ys, Xs, Zs, ZX, ZY,
        Winv, Winvfac, resids, H, WZY, XZWZY, WZX, XZWZX, G, WG, Wup, vce)
end

"""
    fit(::Type{<:LinearCUGMM}, solvertype, vce::CovarianceEstimator, data, eqs; kwargs...)

Conduct linear continuous-updating GMM estimation with
a solver of `solvertype` and a variance-covariance matrix estimated by `vce`.
`data` is a `Tables.jl`-compatible data table.
`eqs` specifies the names of the variables.
This method is only relevant for the case where the parameters are over-identified.
See documentation website for details.

# Keywords
- `nocons::Bool=false`: do not add constant terms automatically.
- `θ0=:TSLS`: initial value of `θ` for the solver to get started; if not specified, the two-stage least squares estimation is conducted to set `θ0`.
- `initonly::Bool=false`: initialize the returned object without conducting the estimation.
- `solverkwargs=NamedTuple()`: keyword arguments passed to the optimization solver as a `NamedTuple`.
- `TF::Type=Float64`: type of the numerical values.
"""
function fit(::Type{<:LinearCUGMM}, solvertype, vce::CovarianceEstimator, data, eqs;
        nocons::Bool=false, θ0=:TSLS, initonly::Bool=false,
        solverkwargs=NamedTuple(), TF::Type=Float64)
    Tables.istable(data) ||
        throw(ArgumentError("data must be a Tables.jl-compatible table"))
    nobs = length(Tables.rows(data))
    eqs, params = _parse_eqs(eqs, nocons)
    est = LinearCUGMM(eqs, nobs, vce; TF=TF)
    nparam = size(est.ZX, 2)
    vcov = Matrix{TF}(undef, nparam, nparam)
    _filldata!(est.Ys, est.Xs, est.Zs, est.ZX, est.ZY, eqs, data)
    coef = Vector{TF}(undef, nparam)
    if θ0 == :TSLS
        fill!(est.Winv, 0)
        r0 = 0
        for Z in est.Zs
            ir = r0+1:r0+size(Z,2)
            mul!(view(est.Winv, ir, ir), Z', Z)
            r0 += size(Z, 2)
        end
        est.Winv ./= nobs # Optional scaling that doesn't matter
        est.Winvfac[] = cholesky!(Hermitian(est.Winv))
        ldiv!(est.WZY, est.Winvfac[], est.ZY)
        mul!(est.XZWZY, est.ZX', est.WZY)
        ldiv!(est.WZX, est.Winvfac[], est.ZX)
        mul!(est.XZWZX, est.ZX', est.WZX)
        ldiv!(coef, cholesky!(Hermitian(est.XZWZX)), est.XZWZY)
    else
        est.Winvfac[] = cholesky!(diagm(0=>ones(nmoment(est))))
        # Expect θ0 to be a vector
        copyto!(coef, θ0)
    end
    solver = _initsolver(solvertype, est, nothing, nothing, nothing, nothing, coef;
        solverkwargs...)
    m = NonlinearGMM(coef, vcov, nothing, nothing, nothing, nothing, est, vce, solver, params)
    initonly || fit!(m)
    return m
end

function (f::VectorObjValue{<:LinearCUGMM})(F, θ)
    est = f.est
    setH!(est.H, est.resids, est.Ys, est.Xs, est.Zs, θ, est.eqs)
    setS!(est.vce, est.H')
    sum!(est.G, est.H')
    est.G ./= size(est.H, 1)
    copyto!(est.Winv, est.vce.S)
    inv!(cholesky!(est.Winv))
    W1 = est.Winvfac[].factors
    copyto!(W1, est.Winv)
    est.Winvfac[] = Wch = cholesky!(Hermitian(W1))
    copyto!(est.Wup, Wch.UL)
    mul!(F, est.Wup, est.G)
end

# Required by the NLopt ext
function (f::ObjValue{<:LinearCUGMM})(θ)
    est = f.est
    setH!(est.H, est.resids, est.Ys, est.Xs, est.Zs, θ, est.eqs)
    setS!(est.vce, est.H')
    copyto!(est.Winv, est.vce.S)
    est.Winvfac[] = cholesky!(Hermitian(est.Winv))
    sum!(est.G, est.H')
    est.G ./= size(est.H, 1)
    ldiv!(est.WG, est.Winvfac[], est.G)
    est.Q[] = est.G'est.WG
end

function setvcov!(m::NonlinearGMM{<:LinearCUGMM})
    est = m.est
    # Preserve the W used for point estimate
    WG = ldiv!(m.vce.vcovcache1, est.Winvfac[], est.ZX)
    GWG = mul!(m.vcov, est.ZX', WG)
    # Cannot directly use WG' as adjoint matrix is not allowed with ldiv!
    GW = copyto!(m.vce.vcovcache2, WG')
    GWGGW = ldiv!(cholesky!(Hermitian(GWG)), GW)
    mul!(m.vce.vcovcache1, m.vce.S, GWGGW')
    mul!(m.vcov, GWGGW, m.vce.vcovcache1)
    m.vcov ./= nobs(m)
end

function fit!(m::NonlinearGMM{<:LinearCUGMM}; kwargs...)
    est = m.est
    solve!(m.solver)
    copyto!(m.coef, m.solver.x)
    # Last evaluation may not be at coef if the trial is rejected
    setH!(est.H, est.resids, est.Ys, est.Xs, est.Zs, m.coef, est.eqs)
    setS!(m.vce, est.H')
    copyto!(est.Winv, m.vce.S)
    est.Winvfac[] = cholesky!(Hermitian(est.Winv))
    sum!(est.G, est.H')
    est.G ./= size(est.H, 1)
    ldiv!(est.WG, est.Winvfac[], est.G)
    est.Q[] = est.G'est.WG
    try
        setvcov!(m)
    catch
        @warn "variance-covariance matrix is not computed"
    end
    return m
end

Jstat(est::Union{<:CUGMM, <:LinearCUGMM}) =
    nmoment(est) > nparam(est) ? nobs(est) * est.Q[] : NaN

function show(io::IO, ::MIME"text/plain", est::Union{<:CUGMM, <:LinearCUGMM}; kwargs...)
    print(io, "Continuously updated GMM estimator")
    mk = nmoment(est) - nparam(est)
    if mk > 0
        J = Jstat(est)
        pv = chisqccdf(mk, J)
        print(io, ":\n    Jstat = ", TestStat(J),
                "        Pr(>J) = ", PValue(pv))
    end
end
