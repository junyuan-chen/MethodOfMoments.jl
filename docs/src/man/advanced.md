# Advanced Usage

For nonlinear problems requiring complicated moment conditions
or a large number of repetitions,
the [basic interface](@ref gdg) based on specifying the `g` and `dg` functions
for individual observations may be suboptimal.
In such circumstances,
it is sufficient to customize the following methods
in order to attain enhanced performance:

| Method | Purpose |
| :--- | :--- |
| `setG!(est, g, θ)` | Update the mean of the residual vectors from the moment conditions |
| `setGH!(est, g, θ)` | Update the individual residuals in addition to the mean updated by `setG!` |
| `setdG!(est, dg, θ)` | Update the mean of the Jacobian matrices of residuals w.r.t. parameters |

Above, `est` is a GMM estimator;
`g` and `dg` are the user-defined objects passed from the basic interface via `fit`;
`θ` is a `Vector` for the current candidate of parameter estimate.
Details on how the computed values are stored can be found
by inspecting the source code of the default implementation.
Some estimators do not use `setG!` and only require `setGH!` and `setdG!`.

Users defining customized methods should restrict the relevant type of `est`
and the customized types for `g` and `dg`.
To illustrate, one should have something like:
```julia
function MethodOfMoments.setG!(est::AbstractGMMEstimator{Nothing,Float64,false}, g::MyG, θ)
    # My customized implementation goes here
end
```

!!! info

    `setG!`, `setGH!` and `setdG!` are the only functions
    making use of `g` or `dg` for the estimation.
    With a customized implementation of them,
    `g` or `dg` can be repurposed as cache for intermediate results
    instead of functions returning the computational results.

For `setGH!`, where individual residuals need to be stored,
performance depends on the shape of the residual matrix.
For the nonlinear estimators,
the residual matrix is *horizontal* by default.
Namely, residual vectors from each observation are stacked as columns in a wide matrix.
If this layout is not ideal for a customized implementation,
one may switch to a vertical layout by simply passing
`Val(false)` to the `horizontal` keyword argument of `fit`.
In that case, residual vectors are expected to be stacked as rows in a tall matrix.
All other functions that interact with the residual matrix
will adjust their behavior accordingly.
