using Test
using MethodOfMoments

using AdvancedMH
using CSV
using DataFrames: DataFrame, combine, groupby, nrow, dropmissing!
using Distributions
using DistributionsAD
using ForwardDiff
using GroupedArrays
using LinearAlgebra
using LogDensityProblems: LogDensityOrder, capabilities, dimension, logdensity,
    logdensity_and_gradient
using MCMCChains
using MethodOfMoments: VarName, PartitionedGMMTasks, checksolvertype, acceptance_rate,
    _default_ntasks, _parse_params, _parse_eqs, _parse_bayes_params, _parse_deriv
using NLopt
using Printf
using StaticArrays
using TransformVariables: as, transform
using TypedTables: Table

# Copying to avoid SentinelArrays that cause allocations with row iteration
exampledata(name::Union{Symbol,String}) =
    DataFrame(CSV.read(MethodOfMoments.datafile(name), DataFrame), copycols=true)

const tests = [
    "iteratedgmm",
    "lineargmm",
    "cugmm",
    "bayesian"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
