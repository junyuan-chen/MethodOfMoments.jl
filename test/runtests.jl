using Test
using MethodOfMoments

using AdvancedMH
using CSV
using DataFrames
using Distributions
using DistributionsAD
using ForwardDiff
using GroupedArrays
using LinearAlgebra
using LogDensityProblems: LogDensityOrder, capabilities, dimension, logdensity,
    logdensity_and_gradient
using MCMCChains
using MethodOfMoments: acceptance_rate
using NLopt
using Printf
using StaticArrays
using TypedTables: Table

# Copying to avoid SentinelArrays that cause allocations with row iteration
exampledata(name::Union{Symbol,String}) =
    DataFrame(CSV.read(MethodOfMoments.datafile(name), DataFrame), copycols=true)

const tests = [
    "iteratedgmm",
    "lineargmm",
    "bayesian"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
