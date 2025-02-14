module MethodOfMoments

using Base: RefValue
using Combinatorics: combinations
using Distributions: Distribution, logpdf
using GroupedArrays: GroupedArray, GroupedVector
using LinearAlgebra: I, mul!, ldiv!, inv!, Hermitian, Cholesky, cholesky!, cholesky, lu!
using LogDensityProblems: LogDensityOrder
using MacroTools
using NonlinearSystems: NonlinearSystem, init, solve!, Hybrid, LeastSquares
using Printf
using StatsAPI: StatisticalModel
using StatsBase: CovarianceEstimator, CoefTable, TestStat, PValue
using StatsFuns: norminvcdf, normccdf, chisqccdf
using Tables
using TransformVariables: AbstractTransform
using TransformedLogDensities: TransformedLogDensity

import Base: show, iterate, parent, getindex
import LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient
import StatsAPI: coef, coefnames, vcov, stderror, confint, nobs, fit, fit!
import StatsBase: mode
import TransformVariables: transform

# Reexport objects from NonlinearSystems
export Hybrid
# Reexport objects from StatsAPI
export coef, coefnames, vcov, stderror, confint, nobs, fit, fit!
# Reexport objects from StatsBase
export mode

export @gdg,

       AbstractGMMResult,
       nparam,
       nmoment,
       Jstat,
       NonlinearGMM,
       LinearGMM,

       RobustVCE,
       ClusterVCE,

       IteratedGMM,

       IteratedLinearGMM,
       JustIdentifiedLinearGMM,

       BayesianGMM,
       logprior,
       loglikelihood!,
       logposterior!,
       logposterior_and_gradient!

include("utils.jl")
include("interface.jl")
include("vce.jl")
include("iteratedgmm.jl")
include("lineargmm.jl")
include("bayesian.jl")

end # module MethodOfMoments
