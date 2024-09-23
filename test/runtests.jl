using Test
using MethodOfMoments

using CSV
using DataFrames
using GroupedArrays
using NLopt
using StaticArrays
using TypedTables: Table

# Copying to avoid SentinelArrays that cause allocations with row iteration
exampledata(name::Union{Symbol,String}) =
    DataFrame(CSV.read(MethodOfMoments.datafile(name), DataFrame), copycols=true)

const tests = [
    "iteratedgmm"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
