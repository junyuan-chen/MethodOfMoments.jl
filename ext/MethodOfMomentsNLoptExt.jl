module MethodOfMomentsNLoptExt

using NLopt
using LinearAlgebra: Hermitian, cholesky!, cholesky, inv!
using MethodOfMoments

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
    MM.setG!(m.est, m.g, m.coef)
    # Solver does not update dG in every step
    MM.setdG!(m.est, m.dg, m.coef)
    MM.setS!(m.vce, m.est.H)
    m.est.iter[] += 1
    return m, state+1
end

end # module
