module MethodOfMomentsForwardDiffExt

using DistributionsAD
using ForwardDiff
using ForwardDiff: DiffResults, derivative!
using MethodOfMoments

const MM = MethodOfMoments

function (p::MM.LogPrior{<:Any,Val{:ForwardDiff}})(θ::TF) where TF
    r = DiffResults.DiffResult(one(TF), one(TF))
    r = derivative!(r, MM.LogPdf(p.d), θ)
    return r.value, r.derivs[1]
end

end # module
