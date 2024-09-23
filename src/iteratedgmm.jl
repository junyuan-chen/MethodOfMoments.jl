struct IteratedGMMTasks{TF}
    rowcuts::Vector{Int}
    Gs::Vector{Vector{TF}}
    dGs::Vector{Matrix{TF}}
end

struct IteratedGMM{P,TF} <: AbstractEstimator{TF}
    iter::RefValue{Int}
    Q::RefValue{TF}
    θlast::Vector{TF}
    diff::RefValue{TF}
    H::Matrix{TF}
    G::Vector{TF}
    WG::Vector{TF}
    dG::Matrix{TF}
    W::Matrix{TF}
    Wfac::Ref{Cholesky{TF,Matrix{TF}}}
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
    Wfac = Ref{Cholesky{TF,Matrix{TF}}}()
    Wup = similar(W)
    ntasks = min(ntasks, nobs)
    if ntasks > 1
        step = nobs ÷ ntasks
        rowcuts = Int[(1:step:1+step*(ntasks-1))..., nobs+1]
        Gs = [Vector{TF}(undef, nmoment) for _ in 1:ntasks]
        dGs = [Matrix{TF}(undef, nmoment, nparam) for _ in 1:ntasks]
        p = IteratedGMMTasks(rowcuts, Gs, dGs)
    else
        p = nothing
    end
    return IteratedGMM(Ref(0), Ref(Inf), θlast, Ref(Inf), H, G, WG, dG, W, Wfac, Wup, p)
end

nparam(est::IteratedGMM) = size(est.dG, 2)
nmoment(est::IteratedGMM) = length(est.G)

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
    if dg === nothing
        return init(Hybrid{LeastSquares}, f, θ0, nmoment; warn=warn, kwargs...)
    else
        j = VectorObjJacobian(est, dg, predg)
        return init(Hybrid{LeastSquares}, f, j, θ0, nmoment; warn=warn, kwargs...)
    end
end

function fit(::Type{<:IteratedGMM}, solvertype, vce::CovarianceEstimator,
        g, params, nmoment::Integer, nobs::Integer;
        dg=nothing, preg=nothing, predg=nothing,
        winitial=I, θtol::Real=1e-8, maxiter::Integer=10000,
        ntasks::Integer=_default_ntasks(nobs*nmoment),
        showtrace::Bool=false,
        initonly::Bool=false, solverkwargs=NamedTuple(), TF::Type=Float64)
    checksolvertype(solvertype)
    params, θ0 = _parse_params(params)
    nparam = length(params)
    est = IteratedGMM(nparam, nmoment, nobs, ntasks; TF=TF)
    # Must initialize W before initializing solver
    copyto!(est.W, winitial)
    est.Wfac[] = cholesky(Hermitian(est.W))
    # solver obj and jac are handled within _initsolver
    solver = _initsolver(solvertype, est, g, dg, preg, predg, θ0; solverkwargs...)
    coef = copy(θ0)
    vcov = Matrix{TF}(undef, nparam, nparam)
    m = NonlinearGMM(coef, vcov, g, dg, est, vce, solver, params)
    initonly || fit!(m; winitial=winitial, θtol=θtol, maxiter=maxiter, showtrace=showtrace)
    return m
end

function setG!(est::IteratedGMM{Nothing,TF}, g, θ) where TF
    N = size(est.H,2)
    fill!(est.G, zero(TF))
    for r in 1:N
        h = g(θ, r)
        est.H[:,r] .= h
        est.G .+= h
    end
    est.G ./= N
end

function setG!(est::IteratedGMM{<:IteratedGMMTasks,TF}, g, θ) where TF
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

function setdG!(est::IteratedGMM{Nothing,TF}, dg, θ) where TF
    N = size(est.H,2)
    fill!(est.dG, zero(TF))
    for r in 1:N
        est.dG .+= dg(θ, r)
    end
    est.dG ./= N
end

function setdG!(est::IteratedGMM{<:IteratedGMMTasks,TF}, dg, θ) where TF
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
    return est.G'est.WG
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
        copyto!(m.est.W, m.vce.S)
        inv!(cholesky!(m.est.W))
        mul!(m.vce.vcovcache1, m.est.W, m.est.dG)
        mul!(m.vcov, m.est.dG', m.vce.vcovcache1)
        inv!(cholesky!(Hermitian(m.vcov)))
    else
        # Preserve the W used for point estimate
        WG = mul!(m.vce.vcovcache1, m.est.W, m.est.dG)
        GWG = mul!(m.vcov, m.est.dG', WG)
        # Cannot directly use WG' as adjoint matrix is not allowed with ldiv!
        GW = copyto!(m.vce.vcovcache2, WG')
        GWGGW = ldiv!(cholesky!(Hermitian(GWG)), GW)
        mul!(m.vce.vcovcache1, m.vce.S, GWGGW')
        mul!(m.vcov, GWGGW, m.vce.vcovcache1)
    end
    m.vcov ./= size(m.est.H,2)
end

function iterate(m::NonlinearGMM{<:IteratedGMM,VCE,<:NonlinearSystem},
        state=1) where VCE
    if state > 1
        copyto!(m.est.W, m.vce.S)
        inv!(cholesky!(m.est.W))
        # Only use the decomposed "half" W
        # Factorization of W for the first iteration is done in _initsolver
        W1 = m.est.Wfac[].factors
        copyto!(W1, m.est.W)
        m.est.Wfac[] = Wch = cholesky!(Hermitian(W1))
        copyto!(m.est.Wup, Wch.UL)
    end
    copyto!(m.est.θlast, m.coef)
    state == 1 ? solve!(m.solver) : solve!(m.solver, m.solver.x)
    copyto!(m.coef, m.solver.x)
    # Last evaluation may not be at coef if the trial is rejected
    setG!(m.est, m.g, m.coef)
    # Solver does not update dG in every step
    setdG!(m.est, m.dg, m.coef)
    mul!(m.est.WG, m.est.W, m.est.G)
    m.est.Q[] = m.est.G'm.est.WG
    setS!(m.vce, m.est.H)
    m.est.iter[] += 1
    return m, state+1
end

@inline function test_θtol!(est::IteratedGMM, θ, tol)
    diff = 0.0
    @inbounds for i in eachindex(θ)
        d = abs(θ[i] - est.θlast[i])
        d > diff && (diff = d)
    end
    est.diff[] = diff
    return diff < tol
end

function _show_trace(io::IO, est::IteratedGMM, newline::Bool, twolines::Bool)
    print(io, "  iter ", lpad(est.iter[], 3), "  =>  ")
    mk = nmoment(est)-nparam(est)
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

Jstat(est::IteratedGMM) =
    nmoment(est) > nparam(est) ? size(est.H, 2) * est.Q[] : NaN

show(io::IO, ::MIME"text/plain", est::IteratedGMM; twolines::Bool=false) =
    (println(io, "Iterated GMM estimator:"); print(io, "  ");
        _show_trace(io, est, false, twolines))
