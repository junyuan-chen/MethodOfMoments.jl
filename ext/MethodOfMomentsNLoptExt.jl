module MethodOfMomentsNLoptExt

using LinearAlgebra: Hermitian, cholesky!, cholesky, inv!
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

function MM._initsolver(opt::NLopt.Opt, est::IteratedGMM, g, dg, preg, predg, θ0; kwargs...)
    f = MM.ObjValue(est, g, preg)
    df = dg === nothing ? nothing : MM.ObjGradient(est, dg, predg)
    fdf = NLoptObj(f, df)
    obj(θ, grad) = fdf(θ, grad)
    NLopt.min_objective!(opt, obj)
    return opt
end

function MM.iterate(m::NonlinearGMM{<:IteratedGMM,VCE,NLopt.Opt}, state=1) where VCE
    if state > 1
        copyto!(m.est.W, m.vce.S)
        inv!(cholesky!(m.est.W))
    end
    copyto!(m.est.θlast, m.coef)
    NLopt.optimize!(m.solver, m.coef)
    # Last evaluation may not be at coef if the trial is rejected
    m.preg === nothing || m.preg(m.coef)
    MM.setG!(m.est, m.g, m.coef)
    # Solver does not update dG in every step
    m.predg === nothing || m.predg(m.coef)
    MM.setdG!(m.est, m.dg, m.coef)
    MM.setS!(m.vce, m.est.H)
    m.est.iter[] += 1
    return m, state+1
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
