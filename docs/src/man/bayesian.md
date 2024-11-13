# Bayesian Quasi-Likelihood

Estimation of nonlinear GMM involves the use of optimization solvers.
An alternative approach that circumvents the need for such solvers
treats the criterion function as a quasi-likelihood function
for tracing out quasi-posterior functions of the parameters,
using a Markov Chain Monte Carlo (MCMC) method [ChernozhukH03A](@citep).
Specifically, we multiply the criterion function shown in
[Generalized Method of Moments](@ref) by ``-\frac{1}{2}``
to obtain the quasi-likelihood.

[MethodOfMoments.jl](https://github.com/junyuan-chen/MethodOfMoments.jl)
provides support for incorporating the quasi-likelihood evaluation in a MCMC sampler
by implementing the
[LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface.
This allows the users to leverage readily available MCMC samplers
from the Julia ecosystem without the need to
defining the quasi-likelihood functions from scratch.
For example, packages such as
[AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl)
for Metropolis-Hastings algorithms recognize
the capability of [MethodOfMoments.jl](https://github.com/junyuan-chen/MethodOfMoments.jl)
for evaluating the log-density.

## Example: Defining Log-Density for MCMC

We reuse the data and specifications from
[Example: Exponential Regression with Instruments](@ref).

```@setup bayesian
using MethodOfMoments, CSV, DataFrames, TypedTables, StaticArrays
# exampledata loads data from CSV files bundled with MethodOfMoments.jl
exampledata(name::Union{Symbol,String}) =
    DataFrame(CSV.read(MethodOfMoments.datafile(name), DataFrame), copycols=true)

data = Table(exampledata(:docvisits))

@gdg function (g::g_stata_gmm_ex8)(θ, r)
    @inbounds d = g.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    z = SVector{7,Float64}((d.private, d.chronic, d.female, d.age, d.black, d.hispanic, 1.0))
    return (d.docvis - exp(θ'x)) .* z
end

g = g_stata_gmm_ex8(data)

@gdg function (dg::dg_stata_gmm_ex8)(θ, r)
    @inbounds d = dg.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    z = SVector{7,Float64}((d.private, d.chronic, d.female, d.age, d.black, d.hispanic, 1.0))
    return z .* (- exp(θ'x) .* x')
end

dg = dg_stata_gmm_ex8(data)

vce = RobustVCE(5, 7, length(data))
```
```@example bayesian
# Assume objects from previous example are already defined
using Distributions
params = (:private=>Uniform(-1,2), :chronic=>Uniform(-1,2),
    :female=>Uniform(-1,2), :income=>Uniform(-1,2), :cons=>Normal())
m = BayesianGMM(vce, g, dg, params, 7, length(data))
```

Above, we provide the names of each parameter and their prior distributions
using distributions defined in
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl).
A `BayesianGMM` contains the ingredients required for computing the log-posterior:
```@example bayesian
θ = [0.5, 1, 1, 0.1, 0.1]
logposterior!(m, θ)
```

To run a Metropolis-Hastings sampler, we may proceed as follows:

```@example bayesian
using AdvancedMH, MCMCChains, LinearAlgebra

spl = MetropolisHastings(RandomWalkProposal{true}(MvNormal(zeros(5), 0.5*I(5))))
N = 10_000
chain = sample(m, spl, N, init_params=θ, param_names=m.params, chain_type=Chains)
```
