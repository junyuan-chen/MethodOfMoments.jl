module MethodOfMomentsStaticArraysExt

using MethodOfMoments
using StaticArrays: SVector

const MM = MethodOfMoments

MM.loglikelihood!(m::BayesianGMM, θ::NamedTuple) =
    loglikelihood!(m, SVector{length(θ)}((θ...,)))

end # module
