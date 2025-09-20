# Linear GMM

An explicit solution for minimizing the criterion function is available
when the moment conditions take the special form
```math
\mathrm{E}\left[(Y_i - \theta'\mathbf{X}_i) \mathbf{Z}_i\right] = \mathbf{0}
```
where the residuals from linear equations are orthogonal to a vector of instruments.
In this case, an optimization solver is unnecessary for the estimation.
If the parameters are just-identified, we have
```math
\hat{\theta} = (\mathbf{Z}'\mathbf{X})^{-1}\mathbf{Z}'\mathbf{Y}
```
where ``\mathbf{Z}``, ``\mathbf{X}`` and ``\mathbf{Y}`` are matrices
formed by stacking ``\mathbf{Z}_i'``, ``\mathbf{X}_i'`` and ``Y_i'`` across observations.
The parameter estimates can be obtained in one step
by directly solving the linear problems.
They are the IV estimates from linear regressions.
If the parameters are over-identified, we have
```math
\hat{\theta} = (\mathbf{X}'\mathbf{Z}\mathbf{W}\mathbf{Z}'\mathbf{X})^{-1}
\mathbf{X}'\mathbf{Z}\mathbf{W}\mathbf{Z}'\mathbf{Y}
```
where ``\mathbf{W}`` is a weight matrix.
The parameter estimates can be obtained iteratively
as for the general nonlinear scenario.
However, we directly evaluate ``\hat{\theta}`` in each iteration.
The two-stage least squares (2SLS) estimator
can be viewed as a special case where the weight matrix is specified as
```math
\mathbf{W} = \left(\mathbf{Z}'\mathbf{Z}\right)^{-1}
```
with no further iteration conducted upon the first evaluation.

## Example: Two-Stage Least Squares

We replicate [Example 2](https://www.stata.com/manuals/rgmm.pdf)
from Stata manual for `gmm`:

```@example lineargmm
using MethodOfMoments, CSV, DataFrames
# exampledata loads data from CSV files bundled with MethodOfMoments.jl
exampledata(name::Union{Symbol,String}) =
    DataFrame(CSV.read(MethodOfMoments.datafile(name), DataFrame), copycols=true)

data = exampledata(:hsng2)
vce = RobustVCE(3, 6, size(data,1))
eq = (:rent, (:hsngval, :pcturban), (:pcturban, :faminc, Symbol.(:reg, 2:4)...))
r = fit(IteratedLinearGMM, vce, data, eq, maxiter=1)
```

Notice that we have specified the estimator by passing `IteratedLinearGMM`
as the first argument.
However, for 2SLS estimation, we restrict `maxiter=1` to avoid further iterations.
The regression equation is specified as a `Tuple` of three elements
where the first one is the column name of the outcome variable in `data`.
The second and third elements are the names of endogenous variables and IVs respectively.
Unless a keyword of `nocons` is specified,
a constant term named `cons` is added automatically for the endogenous variables and IVs.
The default initial weight matrix for `IteratedLinearGMM`
is the one that gives us 2SLS estimates.

!!! warning

    The implementation of the estimator does not handle multicollinearity issues.
    In case some of the regressors are highly correlated,
    the estimator may fail to generate a result.

## [Example: GMM IV Estimation with Clustering](@id GMMIVexample)

We replicate [Example 4](https://www.stata.com/manuals/rivregress.pdf)
from Stata manual for `ivregress`:

```@example lineargmm
data = exampledata(:nlswork)
data[!,:age2] = data.age.^2
dropmissing!(data, [:ln_wage, :age, :birth_yr, :grade, :tenure, :union, :wks_work, :msp])
vce = ClusterVCE(data, :idcode, 6, 8)
eq = (:ln_wage, (:tenure=>[:union, :wks_work, :msp], :age, :age2, :birth_yr, :grade))
r = fit(IteratedLinearGMM, vce, data, eq, maxiter=2)
```

Here we make use of the cluster-robust VCE via `ClusterVCE`
and specify `idcode` as the variable that identifies the clusters.
The regression equation is specified in an alternative format
with a `Tuple` only containing two elements.
The first element is still the name of the outcome variable.
The second element contains the names of all the regressors,
with the endogenous variable `tenure` being paired with a vector of IVs.

## Example: System of Simultaneous Equations

It is possible to specify a system of equations that are estimated jointly.
We consider a modified [Example 16](https://www.stata.com/manuals/rgmm.pdf)
from Stata manual for `gmm`,
replacing the variance-covariance estimator with `RobustVCE`:[^1]

[^1]: Identical estimates can be produced in Stata by changing the `wmatrix` option to `wmatrix(robust)`.

```@example lineargmm
data = exampledata(:klein)
vce = RobustVCE(7, 8, nrow(data))
eqs = [(:consump, (:wagepriv, :wagegovt), (:wagegovt, :govt, :capital1)),
    (:wagepriv, (:consump, :govt, :capital1), (:wagegovt, :govt, :capital1))]
r = fit(IteratedLinearGMM, vce, data, eqs, maxiter=2)
```

For multiple equations, we simply need to collect
the specification for each equation in a vector.
The name of each parameter is prepended by its corresponding outcome variable
in order to distinguish the equation it belongs to.

## Example: Just-Identified IV Regression

Lastly, we illustrate the use of the specialized estimator `JustIdentifiedLinearGMM`
for the just-identified linear GMM using the `auto` dataset from Stata:

```@example lineargmm
data = exampledata(:auto)
vce = RobustVCE(3, 3, size(data,1))
eq = (:mpg, [:weight, :length=>:trunk])
r = fit(JustIdentifiedLinearGMM, vce, data, eq)
```

!!! note

    The familiar OLS regression is estimated if we omit the IV in this example.
