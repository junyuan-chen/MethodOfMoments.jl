struct PartitionedGMMTasks{TF}
    rowcuts::Vector{Int}
    Gs::Vector{Vector{TF}}
    dGs::Vector{Matrix{TF}}
end

struct IteratedGMM{P,TF} <: AbstractGMMEstimator{P,TF}
    iter::RefValue{Int}
    Q::RefValue{TF}
    θlast::Vector{TF}
    diff::RefValue{TF}
    H::Matrix{TF}
    G::Vector{TF}
    WG::Vector{TF}
    dG::Matrix{TF}
    W::Matrix{TF}
    Wfac::RefValue{Cholesky{TF,Matrix{TF}}}
    Wup::Matrix{TF} # Don't use UpperTriangular/LowerTriangular (answer differs)
    p::P
end

function IteratedGMM(nparam::Integer, nmoment::Integer, nobs::Integer, ntasks::Integer;
        TF::Type=Float64)
    θlast = Vector{TF}(undef, nparam)
    # H is horizontal
    H = Matrix{TF}(undef, nmoment, nobs)
    G = Vector{TF}(undef, nmoment)
    WG = Vector{TF}(undef, nmoment)
    dG = Matrix{TF}(undef, nmoment, nparam)
    W = Matrix{TF}(undef, nmoment, nmoment)
    Wfac = RefValue{Cholesky{TF,Matrix{TF}}}()
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
    return IteratedGMM(Ref(0), Ref(NaN), θlast, Ref(NaN), H, G, WG, dG, W, Wfac, Wup, p)
end

nparam(est::IteratedGMM) = size(est.dG, 2)
nmoment(est::IteratedGMM) = length(est.G)

# Fallback method where dg is expected to be a defined function
_initdg(dg::Any, g, params, nmoment) = dg

checksolvertype(T::Type) = throw(ArgumentError("solver of type $T is not supported"))
checksolvertype(::Type{<:Hybrid}) = true

function _initsolver(::Type{<:Hybrid}, est::IteratedGMM, g, dg, preg, predg, θ0;
        warn=false, kwargs...)
    nmoment = length(est.G)
    W1 = est.Wfac[].factors
    copyto!(W1, est.W)
    est.Wfac[] = Wch = cholesky!(Hermitian(W1))
    copyto!(est.Wup, Wch.UL)
    f = VectorObjValue(est, g, preg)
    # Call preg once in case uninitiated values can create NaNs in jacobian
    preg === nothing || preg(θ0)
    j = VectorObjJacobian(est, dg, predg)
    return init(Hybrid{LeastSquares}, f, j, θ0, nmoment; warn=warn, kwargs...)
end

"""
    fit(::Type{<:IteratedGMM}, solvertype, vce::CovarianceEstimator,
        g, dg, params, nmoment::Integer, nobs::Integer; kwargs...)

Conduct nonlinear iterated GMM estimation
with a solver of `solvertype` and weight matrix in each iteration
evaluated as the inverse of variance-covariance matrix estimated by `vce`.
Moment conditions and their derivatives are specified as `g` and `dg`.
Names of the parameters, number of moment conditions
and number of observations are provided as `params`, `nmoment` and `nobs`.
See documentation website for details.

# Keywords
- `preg=nothing`: a function for processing the data frame before evaluating moment conditions.
- `predg=nothing`: a function for processing the data frame before evaluating the derivatives for moment conditions.
- `winitial=I`: initial weight matrix; use identify matrix by default.
- `θtol::Real=1e-8`: tolerance level for determining the convergence of parameters.
- `maxiter::Integer=10000`: maximum number of iterations allowed.
- `ntasks::Integer=_default_ntasks(nobs*nmoment)`: number of threads use for evaluating moment conditions and their derivatives across observations.
- `showtrace::Bool=false`: print information as iteration proceeds.
- `initonly::Bool=false`: initialize the returned object without conducting the estimation.
- `solverkwargs=NamedTuple()`: keyword arguments passed to the optimization solver as a `NamedTuple`.
- `TF::Type=Float64`: type of the numerical values.
"""
function fit(::Type{<:IteratedGMM}, solvertype, vce::CovarianceEstimator,
        g, dg, params, nmoment::Integer, nobs::Integer;
        preg=nothing, predg=nothing,
        winitial=I, θtol::Real=1e-8, maxiter::Integer=10000,
        ntasks::Integer=_default_ntasks(nobs*nmoment),
        showtrace::Bool=false,
        initonly::Bool=false, solverkwargs=NamedTuple(), TF::Type=Float64)
    checksolvertype(solvertype)
    params, θ0 = _parse_params(params)
    nparam = length(params)
    dg = _initdg(dg, g, params, nmoment)
    est = IteratedGMM(nparam, nmoment, nobs, ntasks; TF=TF)
    # Must initialize W before initializing solver
    copyto!(est.W, winitial)
    est.Wfac[] = cholesky(Hermitian(est.W))
    # solver obj and jac are handled within _initsolver
    solver = _initsolver(solvertype, est, g, dg, preg, predg, θ0; solverkwargs...)
    coef = copy(θ0)
    vcov = Matrix{TF}(undef, nparam, nparam)
    m = NonlinearGMM(coef, vcov, g, dg, preg, predg, est, vce, solver, params)
    initonly || fit!(m; winitial=winitial, θtol=θtol, maxiter=maxiter, showtrace=showtrace)
    return m
end

function setG!(est::AbstractGMMEstimator{Nothing,TF}, g, θ) where TF
    N = size(est.H, 2)
    fill!(est.G, zero(TF))
    for r in 1:N
        h = g(θ, r)
        est.H[:,r] .= h
        est.G .+= h
    end
    est.G ./= N
end

function setG!(est::AbstractGMMEstimator{<:PartitionedGMMTasks,TF}, g, θ) where TF
    rowcuts = est.p.rowcuts
    ntasks = length(rowcuts) - 1
    @sync for i in 1:ntasks
        Threads.@spawn begin
            G = est.p.Gs[i]
            fill!(G, zero(TF))
            for r in rowcuts[i]:rowcuts[i+1]-1
                h = g(θ, r)
                est.H[:,r] .= h
                G .+= h
            end
        end
    end
    fill!(est.G, zero(TF))
    for i in 1:ntasks
        est.G .+= est.p.Gs[i]
    end
    est.G ./= rowcuts[end]-1
end

function setdG!(est::AbstractGMMEstimator{Nothing,TF}, dg, θ) where TF
    N = size(est.H, 2)
    fill!(est.dG, zero(TF))
    for r in 1:N
        est.dG .+= dg(θ, r)
    end
    est.dG ./= N
end

function setdG!(est::AbstractGMMEstimator{<:PartitionedGMMTasks,TF}, dg, θ) where TF
    rowcuts = est.p.rowcuts
    ntasks = length(rowcuts) - 1
    @sync for i in 1:ntasks
        Threads.@spawn begin
            dG = est.p.dGs[i]
            fill!(dG, zero(TF))
            for r in rowcuts[i]:rowcuts[i+1]-1
                dG .+= dg(θ, r)
            end
        end
    end
    fill!(est.dG, zero(TF))
    for i in 1:ntasks
        est.dG .+= est.p.dGs[i]
    end
    est.dG ./= rowcuts[end]-1
end

function (f::VectorObjValue{<:IteratedGMM})(F, θ)
    est = f.est
    f.pre === nothing || f.pre(θ)
    setG!(est, f.g, θ)
    # Weight matrix should have been Cholesky decomposed
    mul!(F, est.Wup, est.G)
end

function (j::VectorObjJacobian{<:IteratedGMM})(J, θ)
    est = j.est
    j.pre === nothing || j.pre(θ)
    setdG!(est, j.dg, θ)
    # Weight matrix should have been Cholesky decomposed
    mul!(J, est.Wup, est.dG)
end

function (f::ObjValue{<:IteratedGMM})(θ)
    est = f.est
    f.pre === nothing || f.pre(θ)
    setG!(est, f.g, θ)
    mul!(est.WG, est.W, est.G)
    est.G'est.WG
end

function (j::ObjGradient{<:IteratedGMM})(V, θ)
    est = j.est
    j.pre === nothing || j.pre(θ)
    setdG!(est, j.dg, θ)
    # ObjValue should have already set WG
    mul!(V, est.dG', est.WG)
end

function setvcov!(m::NonlinearGMM{<:IteratedGMM})
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
    # ! H is horizontal
    m.vcov ./= size(est.H, 2)
end

function iterate(m::NonlinearGMM{<:IteratedGMM,VCE,<:NonlinearSystem},
        state=1) where VCE
    est = m.est
    if state > 1
        copyto!(est.W, m.vce.S)
        inv!(cholesky!(est.W))
        # Only use the decomposed "half" W
        # Factorization of W for the first iteration is done in _initsolver
        W1 = est.Wfac[].factors
        copyto!(W1, est.W)
        est.Wfac[] = Wch = cholesky!(Hermitian(W1))
        copyto!(est.Wup, Wch.UL)
        # Manually run remaining parts of f and df for new initial values in solver
        mul!(m.solver.fdf.F, est.Wup, est.G)
        mul!(m.solver.fdf.DF, est.Wup, est.dG)
    end
    copyto!(est.θlast, m.coef)
    state == 1 ? solve!(m.solver) : solve!(m.solver, m.solver.x; initf=false, initdf=false)
    copyto!(m.coef, m.solver.x)
    # Last evaluation may not be at coef if the trial is rejected
    m.preg === nothing || m.preg(m.coef)
    setG!(est, m.g, m.coef)
    # Solver does not update dG in every step
    m.predg === nothing || m.predg(m.coef)
    setdG!(est, m.dg, m.coef)
    mul!(est.WG, est.W, est.G)
    est.Q[] = est.G'est.WG
    setS!(m.vce, est.H)
    est.iter[] += 1
    return m, state+1
end

@inline function test_θtol!(est::AbstractGMMEstimator, θ, tol)
    diff = 0.0
    @inbounds for i in eachindex(θ)
        d = abs(θ[i] - est.θlast[i])
        d > diff && (diff = d)
    end
    est.diff[] = diff
    return diff < tol
end

function _show_trace(io::IO, est::AbstractGMMEstimator, newline::Bool, twolines::Bool)
    print(io, "  iter ", lpad(est.iter[], 3), "  =>  ")
    mk = nmoment(est) - nparam(est)
    @printf(io, "Q(θ) = %11.5e  max|θ-θlast| = %11.5e", est.Q[], est.diff[])
    if mk > 0
        J = Jstat(est)
        pv = chisqccdf(mk, J)
        if twolines
            print(io, "\n                  Jstat = ", TestStat(J),
                "        Pr(>J) = ", PValue(pv))
        else
            print(io, "  Jstat = ", TestStat(J), "  Pr(>J) = ", PValue(pv))
        end
    end
    newline && println(io)
end

function fit!(m::NonlinearGMM{<:IteratedGMM};
        θtol::Real=1e-8, maxiter::Integer=10000, showtrace::Bool=false, kwargs...)
    for iter in 1:maxiter
        iterate(m, iter)
        test_θtol!(m.est, m.coef, θtol) && break
        showtrace && _show_trace(stdout, m.est, true, false)
    end
    showtrace && _show_trace(stdout, m.est, true, false)
    try
        setvcov!(m)
    catch
        @warn "variance-covariance matrix is not computed"
    end
end

# H is horizontal
Jstat(est::IteratedGMM) =
    nmoment(est) > nparam(est) ? size(est.H, 2) * est.Q[] : NaN

show(io::IO, ::MIME"text/plain", est::IteratedGMM; twolines::Bool=false) =
    (println(io, "Iterated GMM estimator:"); print(io, "  ");
        _show_trace(io, est, false, twolines))
