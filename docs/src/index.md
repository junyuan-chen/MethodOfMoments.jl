# MethodOfMoments.jl

Welcome to the documentation site for MethodOfMoments.jl!

[MethodOfMoments.jl](https://github.com/junyuan-chen/MethodOfMoments.jl)
is a Julia package for generalized method of moments (GMM) estimation.
It is designed with performance in mind for estimation involving large datasets.

GMM is a versatile and widely used econometric approach
for parameter estimation in economics and finance.
Notably, the ordinary least squares (OLS) estimator in linear regression
can be viewed as a GMM estimator with a specific form of moment conditions.
More broadly, GMM allows researchers to estimate parameters in theoretical models
without fully specifying the data-generating process,
making it especially valuable when assumptions about the data are hard to defend.

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
