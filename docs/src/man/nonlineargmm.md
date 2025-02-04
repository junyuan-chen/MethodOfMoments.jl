# Generalized Method of Moments

In general, a GMM estimator chooses the point estimate ``\theta`` by
minimizing the following criterion function

```math
Q(\theta) = \left[\frac{1}{N}\sum_{i=1}^N \mathbf{g}_i(\theta)\right]'\mathbf{W}\left[\frac{1}{N}\sum_{i=1}^N \mathbf{g}_i(\theta)\right]
```

where ``\mathbf{g}_i(\theta)`` is a vector of residuals
from evaluating the moment conditions with parameter vector ``\theta``
and observation `i` of a sample consisting of ``N`` observations.
When the number of moment conditions (length of ``\mathbf{g}_i(\theta)``)
is greater than the number of parameters (length of ``\theta``),
the parameters are **over-identified**.
In this case, ``\mathbf{W}`` is a weight matrix
affecting the relative importance of each moment condition for the criterion function.
When the number of moment conditions matches the number of parameters,
the weight matrix is irrelevant and the parameters are **just-identified**.[^1]

[^1]: A suitable rank condition is assumed for discussing the number of moment conditions to be meaningful.

## Nonlinear Iterated GMM Estimator

An iterated GMM estimator is implemented
for some arbitrary moment conditions provided by users.
This allows estimating nonlinear GMM via the iterative method
considered in [HansenHY96F](@citet).
In particular, the **two-step GMM estimator** is a special case
where the number of iterations is restricted to two.

Starting from an initial weight matrix ``\mathbf{W}^{(1)}``,
the iterated GMM estimator iterates through the following steps for ``k\geq 1``:

1. Given a weight matrix ``\mathbf{W}^{(k)}``, find ``\theta^{(k)}`` that minimizes the criterion function.
2. Given the minimizer ``\theta^{(k)}``, compute a new weight matrix ``\mathbf{W}^{(k+1)}`` using the inverse of the variance-covariance matrix estimated from the moment conditions ``\mathbf{g}_i(\theta^{(k)})`` for each ``i``.

The above iterations continue until one of the two conditions are met:

1. ``\theta^{(k+1)}`` is sufficiently close to ``\theta^{(k)}``.
2. A maximum number of iterations is reached.

## Example: Exponential Regression with Instruments

To illustrate the use of such an estimator,
we replicate [Example 8](https://www.stata.com/manuals/rgmm.pdf) from Stata manual for `gmm`.
The data for this example are included in
[MethodOfMoments.jl](https://github.com/junyuan-chen/MethodOfMoments.jl)
for convenience and can be loaded as below:

```@example nonlineargmm
using MethodOfMoments, CSV, DataFrames, TypedTables
# exampledata loads data from CSV files bundled with MethodOfMoments.jl
exampledata(name::Union{Symbol,String}) =
    DataFrame(CSV.read(MethodOfMoments.datafile(name), DataFrame), copycols=true)

data = Table(exampledata(:docvisits))
```

Above, the data frame is converted to a `Table` from
[TypedTables.jl](https://github.com/JuliaData/TypedTables.jl)
for a convenient syntax of evaluating moment conditions by row while being fast.

!!! warning

    It is important to clean the sample before conducting estimation.
    The estimator does not handle invalid values
    such as `missing` and `NaN` when evaluating the moment conditions.

For this example, the moment conditions can be written as
```math
\mathrm{E}[\mathbf{g}_i(\theta)] =
\mathrm{E}\left[ (Y_i - \exp(\theta'\mathbf{X}_i)) \mathbf{Z}_i  \right] = \mathbf{0}
```
where the residual from the structural equation
```math
Y_i = \exp(\theta'\mathbf{X}_i) + \varepsilon_i
```
is assumed to be orthogonal to a vector of instruments ``\mathbf{Z}_i``.

## Specifying Moment Conditions and Their Derivatives

The moment conditions and their derivatives with respect to the parameters
need to be provided as Julia functions
that are going to be evaluated for individual observations.
They are required to accept the parameters as the first argument
and the row index of the data frame as the second argument.
For the moment conditions,
the function should return the residuals
from evaluating the moment conditions for a specific observation
as an iterable object such as a `Tuple` or a static vector `SVector`.
For the derivatives,
the function should return a Jacobian matrix
with rows corresponding to the moment conditions
and columns corresponding to the parameters.

For example, the moment conditions could be defined as follows:

```@example
using StaticArrays

struct g_stata_gmm_ex8{D}
    data::D
end

function (g::g_stata_gmm_ex8)(θ, r)
    @inbounds d = g.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    z = SVector{7,Float64}((d.private, d.chronic, d.female, d.age, d.black, d.hispanic, 1.0))
    return (d.docvis - exp(θ'x)) .* z
end
```

Notice that since such functions typically need to retrieve values from a data frame
that is not an argument of the functions,
a `struct` wrapping the data frame has been defined
and a method is attached to the struct following the requirement for the arguments.

!!! info

    Illustrations for adding a method to a `struct`
    is available [here](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects)
    from Julia documentation.

To simplify the above definition, a macro `@dgd` is provided.
This allows defining a `struct` as in the above example
by prepending `@dgd` to the function definition
without typing the definition for the `struct`:

```@example nonlineargmm
using StaticArrays

@gdg function (g::g_stata_gmm_ex8)(θ, r)
    @inbounds d = g.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    z = SVector{7,Float64}((d.private, d.chronic, d.female, d.age, d.black, d.hispanic, 1.0))
    return (d.docvis - exp(θ'x)) .* z
end

g = g_stata_gmm_ex8(data)
```

!!! info

    Using `SVector` from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
    instead of `Vector` avoids memory allocations
    while preserving the syntax for array operations.

For minimizing the criterion function and computing the variance-covariance matrix,
we additionally require the derivatives of the moment conditions
with respect to each parameter.
They can be defined in a similar fashion as below:

```@example nonlineargmm
@gdg function (dg::dg_stata_gmm_ex8)(θ, r)
    @inbounds d = dg.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    z = SVector{7,Float64}((d.private, d.chronic, d.female, d.age, d.black, d.hispanic, 1.0))
    return z .* (- exp(θ'x) .* x')
end

dg = dg_stata_gmm_ex8(data)
```

!!! warning

    The users are expected to order the parameters and moment conditions consistently.
    An index `k` for a parameter should always refer to the same parameter
    whenever a vector of parameters is involved.
    A similar requirement holds for the moment conditions.

!!! note

    The performance of these functions evaluating the moment conditions
    and their derivatives is important for the run time of the estimator,
    especially when the sample size is large.
    The users are recommended to profile these functions
    to ensure they have acceptable performance.
    Ideally, these functions should be non-allocating and type stable.

## Specifying Variance-Covariance Estimator

A variance-covariance estimator (VCE) provides information for inference.
With the iterated GMM estimator,
the VCE additionally instructs how the weight matrix ``\mathbf{W}^{(k)}``
is updated in each iteration.

[MethodOfMoments.jl](https://github.com/junyuan-chen/MethodOfMoments.jl)
implements two VCEs:

1. The Eicker-Huber-White heteroskedasticity-robust VCE.
2. The multiway cluster-robust VCE as in [CameronGM11R](@citet).

Here, we specify the heteroskedasticity-robust VCE by constructing a `RobustVCE`,
specifying the numbers of parameters, moment conditions and observations as arguments:

```@example nonlineargmm
vce = RobustVCE(5, 7, length(data))
```

## Obtaining the Estimation Results

We are now ready to conduct the estimation:

```@example nonlineargmm
# Specify parameter names
params = (:private, :chronic, :female, :income, :cons)
# Conduct the estimation
r = fit(IteratedGMM, Hybrid, vce, g, dg, params, 7, length(data), maxiter=2, ntasks=2)
```

We have specified the estimator by providing `IteratedGMM` as the first argument.
For minimizing the criterion function,
we have used the `Hybrid` solver from
[NonlinearSystems.jl](https://github.com/junyuan-chen/NonlinearSystems.jl),
which is a native Julia implementation of the hybrid method from MINPACK
with minor revisions.
With the keyword argument `maxiter=2`,
we only conduct two steps for the iteration.
The moment conditions across observations are evaluated in parallel with two threads
specified with `ntasks=2`.
If `ntasks` is not specified,
a default value will be determined based on the sample size.

!!! info

    The `Hybrid` solver from
    [NonlinearSystems.jl](https://github.com/junyuan-chen/NonlinearSystems.jl)
    is the only solver bundled with
    [MethodOfMoments.jl](https://github.com/junyuan-chen/MethodOfMoments.jl)
    at this moment.
    It is possible to swap the solver,
    although that should be unnecessary in typical use cases.

Interface for retrieving the estimation results is defined following
[StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).
For example, to obtain the point estimates:

```@example nonlineargmm
coef(r)
```

To obtain the standard errors:

```@example nonlineargmm
stderror(r)
```

To obtain 90% confidence intervals:

```@example nonlineargmm
lb, ub = confint(r; level=0.9)
```

When the parameters are over-identified,
the Hansen's ``J`` statistic can be retrieved:

```@example nonlineargmm
Jstat(r)
```

## Iterating Step by Step

Results of iterating the GMM estimator for a small number of steps
can be obtained by setting the keyword argument `maxiter`.
Furthermore, there is no need to restart the entire estimation procedure
for different numbers of steps.
Suppose we are interested in the results for
both the one-step and two-step GMM estimators,
we can first obtain the results from the first step:

```@example nonlineargmm
# One-step GMM
r = fit(IteratedGMM, Hybrid, vce, g, dg, params, 7, length(data), maxiter=1)
```

After copying the estimates of interest (e.g., `copy(coef(r))`),
we proceed to the next step using `fit!`
that updates the results in-place:

```@example nonlineargmm
# Two-step GMM without repeating the first step
fit!(r, maxiter=2)
```

Notice that we have reused the same result object `r`
and set a different value for `maxiter`.

!!! info

    The Hansen's ``J`` statistic is not reported for one-step GMM estimators
    (`Jstat` returns `NaN`).
    This is because the initial weight matrix for the GMM criterion can be arbitrary,
    resulting in meaningless scaling for ``Q(\theta)``.

## Implementation Details

Instead of literally solving a minimization problem
defined with the scalar-valued criterion function ``Q(\theta)``,
the estimator solves a least squares problem for
the vector-valued moment conditions after adjusting the weights
using the `Hybrid` solver designed for systems of nonlinear equations.
This relies on the assumption that the weight matrix
should always be positive definite
and hence can be decomposed as a product of a triangular matrix
and its conjugate transpose with Cholesky decomposition.
This approach tends to be much faster than
solving the scalar-valued problem,
as it exploits more information from the Jacobian matrix of the moment conditions
relative to the gradient vector for the scalar-valued problem.
