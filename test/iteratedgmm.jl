@gdg function (g::g_stata_gmm_ex6)(θ, r)
    @inbounds d = g.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    return (d.docvis - exp(θ'x)) .* x
end

@gdg function (dg::dg_stata_gmm_ex6)(θ::Vector, r)
    @inbounds d = dg.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    return x .* (- exp(θ'x) .* x')
end

@gdg function (g::g_stata_gmm_ex8)(θ, r)
    @inbounds d = g.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    z = SVector{7,Float64}((d.private, d.chronic, d.female, d.age, d.black, d.hispanic, 1.0))
    return (d.docvis - exp(θ'x)) .* z
end

@gdg function (dg::dg_stata_gmm_ex8)(θ, r)
    @inbounds d = dg.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    z = SVector{7,Float64}((d.private, d.chronic, d.female, d.age, d.black, d.hispanic, 1.0))
    return z .* (- exp(θ'x) .* x')
end

@gdg function (g::g_stata_gmm_ex8_scaled)(θ, r)
    @inbounds d = g.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    z = SVector{7,Float64}((d.private, d.chronic, d.female, d.age, d.black, d.hispanic, 1.0))
    return (d.docvis - exp(θ'x)) .* z
end

@gdg function (dg::dg_stata_gmm_ex8_scaled)(θ, r)
    @inbounds d = dg.data[r]
    x = SVector{5,Float64}((d.private, d.chronic, d.female, d.income, 1.0))
    z = SVector{7,Float64}((d.private, d.chronic, d.female, d.age, d.black, d.hispanic, 1.0))
    return z .* (- exp(θ'x) .* x') ./ 2000
end

struct preg_stata_gmm_ex11{D}
    data::D
    mubar::Vector{Float64}
    ybar::Vector{Float64}
    Nid::Vector{Int}
end

function (p::preg_stata_gmm_ex11)(θ)
    fill!(p.mubar, 0)
    fill!(p.ybar, 0)
    @inbounds for r in eachindex(p.data)
        d = p.data[r]
        x = SVector{3,Float64}((d.x1, d.x2, d.x3))
        g = Int(d.id)
        mu = exp(θ'x)
        p.data.mu[r] = mu
        p.mubar[g] += mu
        p.ybar[g] += d.y
    end
    p.mubar ./= p.Nid
    p.ybar ./= p.Nid
end

struct g_stata_gmm_ex11{D}
    data::D
    mubar::Vector{Float64}
    ybar::Vector{Float64}
end

function (g::g_stata_gmm_ex11)(θ, r)
    @inbounds d = g.data[r]
    i = Int(d.id)
    x = SVector{3,Float64}((d.x1, d.x2, d.x3))
    return (d.y - d.mu * g.ybar[i] / g.mubar[i]) .* x
end

struct predg_stata_gmm_ex11{D}
    data::D
    dmubar::Matrix{Float64}
    ybar::Vector{Float64}
    Nid::Vector{Int}
end

function (p::predg_stata_gmm_ex11)(θ)
    fill!(p.dmubar, 0)
    @inbounds for r in eachindex(p.data)
        d = p.data[r]
        x = SVector{3,Float64}((d.x1, d.x2, d.x3))
        g = Int(d.id)
        mu = exp(θ'x)
        p.dmubar[1,g] += p.data.dmu1[r] = mu * d.x1
        p.dmubar[2,g] += p.data.dmu2[r] = mu * d.x2
        p.dmubar[3,g] += p.data.dmu3[r] = mu * d.x3
    end
    p.dmubar ./= p.Nid'
end

struct dg_stata_gmm_ex11{D}
    data::D
    mubar::Vector{Float64}
    dmubar::Matrix{Float64}
    ybar::Vector{Float64}
end

function (dg::dg_stata_gmm_ex11)(θ, r)
    @inbounds d = dg.data[r]
    i = Int(d.id)
    x = SVector{3,Float64}((d.x1, d.x2, d.x3))
    dmu = SVector{3,Float64}((d.dmu1, d.dmu2, d.dmu3))
    return (-dg.ybar[i] .* (dmu.*dg.mubar[i] .- d.mu.*view(dg.dmubar,:,i)) ./ (dg.mubar[i])^2)' .* x
end

@testset "IteratedGMM" begin
    data = Table(exampledata(:docvisits))
    params = (:private, :chronic, :female, :income, :cons)

    g = g_stata_gmm_ex6(data)
    dg = dg_stata_gmm_ex6(data)
    vce = RobustVCE(5, 5, length(data))
    r = fit(IteratedGMM, Hybrid, vce, g, dg, params, 5, length(data), ntasks=1,
        solverkwargs=(showtrace=5,))
    # Compare results with Stata
    # gmm (docvis - exp({xb:private chronic female income _cons})),
    #    instruments(private chronic female income) igmm
    @test coef(r) ≈ [0.79866538, 1.0918651, 0.49254807, 0.00355701, -0.22972634] atol=1e-6
    @test vcov(r)[:,1] ≈ [0.01187862, -0.00049431, 0.00081862, -0.00003692, -0.00981242] atol=1e-7
    @test stderror(r) ≈ [0.1089891, 0.0559888, 0.0585298, 0.0010824, 0.1108607] atol=1e-6
    @test coefnames(r) == collect(params)
    @test isnan(Jstat(r))

    @test sprint(show, r.est) == "IteratedGMM"

    @test sprint(show, r) == "5×5 NonlinearGMM"
    str = sprint(show, MIME("text/plain"), r)
    # The Q almost 0 and varies across machines
    Qstr = @sprintf("%11.5e", r.est.Q[])
    @test str[1:789] == """
        NonlinearGMM with 5 moments and 5 parameters:
          Iterated GMM estimator:
            iter   2  =>  Q(θ) = $Qstr  max|θ-θlast| = 8.06983e-11
          Heteroskedasticity-robust covariance estimator
        ───────────────────────────────────────────────────────────────────────────
                    Estimate  Std. Error      z  Pr(>|z|)    Lower 95%    Upper 95%
        ───────────────────────────────────────────────────────────────────────────
        private   0.798665    0.108989     7.33    <1e-12   0.585051     1.01228"""

    g = g_stata_gmm_ex8(data)
    dg = dg_stata_gmm_ex8(data)
    vce = RobustVCE(5, 7, length(data))
    r = fit(IteratedGMM, Hybrid, vce, g, dg, collect(params), 7, length(data), ntasks=2)

    # gmm (docvis - exp({xb:private chronic female income _cons})),
    #    instruments(private chronic female age black hispanic) igmm winitial(identity)
    b = [0.52261931, 1.0880932, 0.67021421, 0.01454361, -0.6036728]
    se = [0.1601102, 0.0622371, 0.0973558, 0.0027007, 0.1387295]
    @test coef(r) ≈ b atol=1e-6
    @test stderror(r) ≈ se atol=1e-6
    # estat overid
    @test Jstat(r) ≈ 8.89575 atol=1e-4

    # Somehow the Linux runner gives a different iter but identical results
    iter = Sys.islinux() ? 16 : 15
    @test sprint(show, MIME("text/plain"), r.est) == """
        Iterated GMM estimator:
            iter  $iter  =>  Q(θ) = 2.01627e-03  max|θ-θlast| = 0.00000e+00  Jstat = 8.90  Pr(>J) = 0.0117"""
    @test sprint(show, MIME("text/plain"), r)[1:821] == """
        NonlinearGMM with 7 moments and 5 parameters:
          Iterated GMM estimator:
            iter  $iter  =>  Q(θ) = 2.01627e-03  max|θ-θlast| = 0.00000e+00
                          Jstat = 8.90        Pr(>J) = 0.0117
          Heteroskedasticity-robust covariance estimator
        ────────────────────────────────────────────────────────────────────────
                   Estimate  Std. Error      z  Pr(>|z|)    Lower 95%  Upper 95%
        ────────────────────────────────────────────────────────────────────────
        private   0.52262    0.16011      3.26    0.0011   0.208809     0.83643"""

    d = data
    Z = [d.private d.chronic d.female d.age d.black d.hispanic ones(length(d))]
    w1 = Z'Z ./ length(d)
    params = params.=> 0.0
    r = fit(IteratedGMM, Hybrid, vce, g, dg, params, 7, length(data), winitial=w1)
    @test coef(r) ≈ b atol=1e-6
    @test stderror(r) ≈ se atol=1e-6

    opt = NLopt.Opt(:LN_BOBYQA, length(params))
    r = fit(IteratedGMM, opt, vce, g, dg, params, 7, length(data))
    @test coef(r) ≈ b atol=1e-6
    @test stderror(r) ≈ se atol=1e-6

    opt = NLopt.Opt(:LN_NELDERMEAD, length(params))
    r = fit(IteratedGMM, opt, vce, g, dg, params, 7, length(data), winitial=w1)
    @test coef(r) ≈ b atol=1e-6
    @test stderror(r) ≈ se atol=1e-6

    # LBFGS doesn't work well and seems to have some scaling issues
    g = g_stata_gmm_ex8_scaled(data)
    dg = dg_stata_gmm_ex8_scaled(data)
    opt = NLopt.Opt(:LD_LBFGS, length(params))
    r = fit(IteratedGMM, opt, vce, g, dg, params, 7, length(data))
    @test coef(r) ≈ b atol=1e-1
    # @test stderror(r) ≈ se atol=1e-2

    data = exampledata(:poisson1)
    for v in (:mu, :dmu1, :dmu2, :dmu3)
        data[!,v] .= 0.0
    end
    mubar = zeros(45)
    dmubar = zeros(3,45)
    ybar = zeros(45)
    Nid = sort!(combine(groupby(data, :id), nrow=>:Nid), :id).Nid
    data = Table(data)
    params = (x1=0.0, x2=0.0, x3=0.0)

    pg = preg_stata_gmm_ex11(data, mubar, ybar, Nid)
    g = g_stata_gmm_ex11(data, mubar, ybar)
    pdg = predg_stata_gmm_ex11(data, dmubar, ybar, Nid)
    dg = dg_stata_gmm_ex11(data, mubar, dmubar, ybar)
    vce = ClusterVCE(data, :id, 3, 3)
    r = fit(IteratedGMM, Hybrid, vce, g, dg, params, 3, length(data), preg=pg, predg=pdg)

    #=
    program gmm_poi
        syntax varlist if, at(name)
        quietly {
        tempvar mu mubar ybar
        generate double `mu' = exp(x1*`at'[1,1] + x2*`at'[1,2] + x3*`at'[1,3]) `if'
        egen double `mubar' = mean(`mu') `if', by(id)
        egen double `ybar' = mean(y) `if', by(id)
        replace `varlist' = y - `mu'*`ybar'/`mubar' `if'
        }
    end
    =#
    # gmm gmm_poi, nequations(1) parameters(b1 b2 b3) instruments(x1 x2 x3, noconstant)
    #    vce(cluster id) igmm
    @test coef(r) ≈ [1.9486602, -2.9661193, 1.0086338] atol=1e-6
    @test stderror(r) ≈ [0.1000265, 0.0923592, 0.1156561] atol=1e-6

    @test sprint(show, MIME("text/plain"), r)[1:697] == """
        NonlinearGMM with 3 moments and 3 parameters:
          Iterated GMM estimator:
            iter   2  =>  Q(θ) = 9.68711e-34  max|θ-θlast| = 1.99001e-11
          Cluster-robust covariance estimator: id
        ────────────────────────────────────────────────────────────────
            Estimate  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
        ────────────────────────────────────────────────────────────────
        x1   1.94866   0.100026    19.48    <1e-83   1.75261     2.14471"""

    @test sprint(show, MIME("text/plain"), vce) == """
        1-way cluster-robust covariance estimator:
          id"""
end
