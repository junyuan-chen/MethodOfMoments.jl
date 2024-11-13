# MethodOfMoments.jl

*Generalized method of moments (GMM) estimation in Julia*

[![CI-stable][CI-stable-img]][CI-stable-url]
[![codecov][codecov-img]][codecov-url]
[![PkgEval][pkgeval-img]][pkgeval-url]
[![docs-stable][docs-stable-img]][docs-stable-url]
[![docs-dev][docs-dev-img]][docs-dev-url]

[CI-stable-img]: https://github.com/junyuan-chen/MethodOfMoments.jl/workflows/CI-stable/badge.svg
[CI-stable-url]: https://github.com/junyuan-chen/MethodOfMoments.jl/actions?query=workflow%3ACI-stable

[codecov-img]: https://codecov.io/gh/junyuan-chen/MethodOfMoments.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/junyuan-chen/MethodOfMoments.jl

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/M/MethodOfMoments.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/M/MethodOfMoments.html

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://junyuan-chen.github.io/MethodOfMoments.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://junyuan-chen.github.io/MethodOfMoments.jl/dev/

[MethodOfMoments.jl](https://github.com/junyuan-chen/MethodOfMoments.jl)
is a Julia package for generalized method of moments (GMM) estimation.
It is designed with performance in mind for estimation involving large datasets.

## Features

- Fast multi-threaded nonlinear iterated GMM estimation
- Specialized methods for linear GMM
- Support for Bayesian quasi-likelihood approach

## Installation

MethodOfMoments.jl can be installed with the Julia package manager
[Pkg](https://docs.julialang.org/en/v1/stdlib/Pkg/).
From the Julia REPL, type `]` to enter the Pkg REPL and run:

```
pkg> add MethodOfMoments
```

For details on usage, please see the [documentation][docs-stable-url].
