module MethodOfMoments

using Base: RefValue
using Combinatorics: combinations
using GroupedArrays: GroupedArray, GroupedVector
using LinearAlgebra: I, mul!, ldiv!, inv!, Hermitian, Cholesky, cholesky!, cholesky
using MacroTools
using NonlinearSystems: NonlinearSystem, init, solve!, Hybrid, LeastSquares
using Printf
using StatsAPI: StatisticalModel
using StatsBase: CovarianceEstimator, CoefTable, TestStat, PValue
using StatsFuns: norminvcdf, normccdf, chisqccdf
using Tables

import Base: show, iterate
export NLopt
import StatsAPI: coef, coefnames, vcov, stderror, confint, fit, fit!

# Reexport objects from StatsAPI
export coef, coefnames, vcov, stderror, confint, fit, fit!
# Reexport objects from NonlinearSystems
export Hybrid

export @gdg,

       NonlinearGMM,
       nparam,
       nmoment,
       Jstat,

       RobustVCE,
       ClusterVCE,

       IteratedGMM

include("utils.jl")
include("interface.jl")
include("vce.jl")
include("iteratedgmm.jl")

end # module MethodOfMoments
