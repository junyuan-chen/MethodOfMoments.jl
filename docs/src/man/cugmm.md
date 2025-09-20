# Continuous-Updating GMM

Instead of taking the weight matrix as given
when evaluating the criterion function,
a **contiuous-updating** (also known as continuously-updated, or CU) GMM estimator
updates the weight matrix simultaneously when the parameter vector ``\theta`` is altered.
Specifically, the criterion function
```math
Q(\theta) = \left[\frac{1}{N}\sum_{i=1}^N \mathbf{g}_i(\theta)\right]'\mathbf{W}(\theta)\left[\frac{1}{N}\sum_{i=1}^N \mathbf{g}_i(\theta)\right]
```
now takes ``\mathbf{W}`` as a function of ``\theta``,
which typically is the inverse of the variance-covariance estimator.
Starting from a guess for the initial value of ``\theta``,
the CUGMM estimator solves an optimization problem
with ``Q(\theta)`` as the objective function.

Implementations for both nonlinear and linear moment conditions are provided
via estimator types `CUGMM` and `LinearCUGMM`.
Clearly, the CUGMM estimator is relevant only
when the number of moment conditions exceeds the number of parameters.

## Example: IV Estimation with CUGMM

We revisit the [GMM IV example](@ref GMMIVexample) with CUGMM:

```@example cugmm
using MethodOfMoments, CSV, DataFrames # hide
exampledata(name::Union{Symbol,String}) = # hide
    DataFrame(CSV.read(MethodOfMoments.datafile(name), DataFrame), copycols=true) # hide
data = exampledata(:nlswork) # hide
data[!,:age2] = data.age.^2 # hide
dropmissing!(data, [:ln_wage, :age, :birth_yr, :grade, :tenure, :union, :wks_work, :msp]) # hide
vce = ClusterVCE(data, :idcode, 6, 8)
eq = (:ln_wage, (:tenure=>[:union, :wks_work, :msp], :age, :age2, :birth_yr, :grade))
r = fit(LinearCUGMM, Hybrid, vce, data, eq)
```

Above, we have used the `Hybrid` solver from
[NonlinearSystems.jl](https://github.com/junyuan-chen/NonlinearSystems.jl).
The moment conditions are specified in the same way as before for [Linear GMM](@ref).

Nonlinear CUGMM estimation is supported with a syntax
similar to that for `IteratedGMM` via `CUGMM`.
