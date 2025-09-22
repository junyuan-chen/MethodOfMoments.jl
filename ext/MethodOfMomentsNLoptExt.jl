module MethodOfMomentsNLoptExt

using LinearAlgebra: Hermitian, cholesky!, cholesky, inv!, mul!, ldiv!
using MethodOfMoments
using MethodOfMoments: BayesOrTrans
using NLopt
using Printf

const MM = MethodOfMoments

MM.checksolvertype(::NLopt.Opt) = true

struct NLoptObj{F,DF}
    f::F
    df::DF
end

function (obj::NLoptObj)(θ, grad)
    # Must evaluate f first for setting WG
    v = obj.f(θ)
    if length(grad) > 0
        # Should not reach here if the algorithm is derivative free
        obj.df(grad, θ)
    end
    return v
end

struct NLoptSolver
    opt::NLopt.Opt
    ret::Base.RefValue{Symbol}
end

function MM._initsolver(opt::NLopt.Opt, est::IteratedGMM, g, dg, preg, predg, θ0; kwargs...)
    f = MM.ObjValue(est, g, preg)
    df = dg === nothing ? nothing : MM.ObjGradient(est, dg, predg)
    fdf = NLoptObj(f, df)
    # NLopt wrapper requires a Function
    obj(θ, grad) = fdf(θ, grad)
    NLopt.min_objective!(opt, obj)
    for (k, v) in kwargs
        setproperty!(opt, k, v)
    end
    haskey(kwargs, :maxeval) || (opt.maxeval = 10_000)
    return NLoptSolver(opt, Base.RefValue{Symbol}())
end

function MM.iterate(m::NonlinearGMM{<:IteratedGMM,VCE,NLoptSolver}, state=1) where VCE
    if state > 1
        copyto!(m.est.W, m.vce.S)
        inv!(cholesky!(m.est.W))
    end
    copyto!(m.est.θlast, m.coef)
    r = NLopt.optimize!(m.solver.opt, m.coef)
    m.solver.ret[] = r[3]
    # Last evaluation may not be at coef if the trial is rejected
    m.preg === nothing || m.preg(m.coef)
    MM.setGH!(m.est, m.g, m.coef)
    # Solver does not update dG in every step
    m.predg === nothing || m.predg(m.coef)
    MM.setdG!(m.est, m.dg, m.coef)
    MM.setS!(m.vce, m.est.H, MM.horizontal(m.est))
    m.est.iter[] += 1
    return m, state+1
end

function MM._initsolver(opt::NLopt.Opt, est::Union{<:CUGMM,<:LinearCUGMM},
        g, dg, preg, predg, θ0; kwargs...)
    f = MM.ObjValue(est, g, preg)
    fdf = NLoptObj(f, nothing)
    obj(θ, grad) = fdf(θ, grad)
    NLopt.min_objective!(opt, obj)
    for (k, v) in kwargs
        setproperty!(opt, k, v)
    end
    haskey(kwargs, :maxeval) || (opt.maxeval = 10_000)
    return NLoptSolver(opt, Base.RefValue{Symbol}())
end

function MM.fit!(m::NonlinearGMM{<:CUGMM,VCE,NLoptSolver}; kwargs...) where VCE<:MM.CovarianceEstimator
    r = NLopt.optimize!(m.solver.opt, m.coef)
    m.solver.ret[] = r[3]
    # Last evaluation may not be at coef if the trial is rejected
    m.preg === nothing || m.preg(m.coef)
    est = m.est
    MM.setGH!(est, m.g, m.coef)
    MM.setS!(est.vce, est.H, MM.horizontal(est))
    copyto!(est.W, est.vce.S)
    inv!(cholesky!(est.W))
    # Solver does not update dG in every step
    m.predg === nothing || m.predg(m.coef)
    MM.setdG!(est, m.dg, m.coef)
    mul!(est.WG, est.W, est.G)
    est.Q[] = est.G'est.WG
    try
        MM.setvcov!(m)
    catch
        @warn "variance-covariance matrix is not computed"
    end
    return m
end

function MM.fit!(m::NonlinearGMM{<:LinearCUGMM,VCE,NLoptSolver}; kwargs...) where VCE<:MM.CovarianceEstimator
    r = NLopt.optimize!(m.solver.opt, m.coef)
    m.solver.ret[] = r[3]
    # Last evaluation may not be at coef if the trial is rejected
    est = m.est
    MM.setH!(est.H, est.resids, est.Ys, est.Xs, est.Zs, m.coef, est.eqs)
    MM.setS!(m.vce, est.H, MM.horizontal(est))
    copyto!(est.Winv, m.vce.S)
    est.Winvfac[] = cholesky!(Hermitian(est.Winv))
    sum!(est.G, est.H')
    est.G ./= size(est.H, 1)
    ldiv!(est.WG, est.Winvfac[], est.G)
    est.Q[] = est.G'est.WG
    try
        MM.setvcov!(m)
    catch
        @warn "variance-covariance matrix is not computed"
    end
    return m
end

# Wrap the objective function for tracing
function nlopt_obj!(m::BayesOrTrans, θ, grad, counter, printgap)
    f = m(θ, grad)
    counter[] += 1
    iter = counter[]
    if printgap > 0 && (iter-1) % printgap == 0
        @printf "iter %4i:  f(θ) = %10f" iter f
        println("  θ = ", θ)
    end
    return f
end

function _printgap(verbose::Union{Bool,Integer})
    if verbose === true
        return 20
    elseif verbose === false
        return 0
    else
        return Int(verbose)
    end
end

function _solve!(m::BayesOrTrans, opt::NLopt.Opt, θ0, maxf::Bool; verbose, kwargs...)
    counter = Ref(0)
    printgap = _printgap(verbose)
    f(x, grad) = nlopt_obj!(m, x, grad, counter, printgap)
    maxf ? (opt.max_objective = f) : (opt.min_objective = f)
    for (k, v) in kwargs
        setproperty!(opt, k, v)
    end
    # NLopt does not impose any stopping criterion by default
    haskey(kwargs, :ftol_abs) || (opt.ftol_abs = 1e-8)
    haskey(kwargs, :maxeval) || (opt.maxeval = 10_000)
    r = NLopt.optimize(opt, θ0)
    r[3] in (:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED) ||
        @warn "NLopt solver status is $(r[3])"
    return r, counter[]
end

function MM.mode(m::BayesOrTrans, solver::NLopt.Opt, θ0::AbstractVector;
        verbose::Union{Bool,Integer}=false, kwargs...)
    r, counter = _solve!(m, solver, θ0, true; verbose, kwargs...)
    # Solver result may not be at the last evaluation left in m
    p = parent(m)
    copyto!(p.coef, m isa BayesianGMM ? r[2] : transform(m.transformation, r[2]))
    return r[2], counter, r
end

end # module
