@testset "IteratedLinearGMM" begin
    data = exampledata(:hsng2)
    vce = RobustVCE(3, 6, size(data,1))
    eq = (:rent, (:hsngval, :pcturban), (:pcturban, :faminc, Symbol.(:reg, 2:4)...))
    r = fit(IteratedLinearGMM, vce, data, eq, maxiter=1)
    # Compare results with Stata GMM Example 2
    # gmm (rent - {xb:hsngval pcturban _cons}),
    #    instruments(pcturban faminc reg2-reg4) vce(unadjusted) onestep
    @test coef(r) ≈ [0.00223983, 0.08151597, 120.70651] atol=1e-5
    @test vcov(r)[:,1] ≈ [4.516e-07, -0.00025511, -0.00353769] atol=1e-8
    @test stderror(r) ≈ [0.000672, 0.4445939, 15.25546] atol=1e-5

    r = fit(IteratedLinearGMM, vce, data, [eq], maxiter=2)
    # gmm (rent - {xb:hsngval pcturban _cons}),
    #    instruments(pcturban faminc reg2-reg4) vce(robust) twostep
    @test coef(r) ≈ [0.00146433, 0.76154816, 112.12271] atol=1e-5
    @test stderror(r) ≈ [0.0004473, 0.2895105, 10.80234] atol=1e-5
    @test r.est.Q[] ≈ 0.13672801 atol=1e-5
    # estat overid
    @test Jstat(r) ≈ 6.8364 atol=1e-3

    @test sprint(show, MIME("text/plain"), r.est) == """
        Iterated Linear GMM estimator:
            iter   2  =>  Q(θ) = 1.36728e-01  max|θ-θlast| = 8.58380e+00  Jstat = 6.84  Pr(>J) = 0.0773"""

    @test sprint(show, MIME("text/plain"), r)[1:898] == """
        LinearGMM with 6 moments and 3 parameters:
          Iterated Linear GMM estimator:
            iter   2  =>  Q(θ) = 1.36728e-01  max|θ-θlast| = 8.58380e+00
                          Jstat = 6.84        Pr(>J) = 0.0773
          Heteroskedasticity-robust covariance estimator
        ─────────────────────────────────────────────────────────────────────────────────
                      Estimate    Std. Error      z  Pr(>|z|)     Lower 95%     Upper 95%
        ─────────────────────────────────────────────────────────────────────────────────
        hsngval     0.00146433   0.000447271   3.27    0.0011   0.000587694    0.00234096"""

    data = exampledata(:nlswork)
    data[!,:age2] = data.age.^2
    dropmissing!(data, [:ln_wage, :age, :birth_yr, :grade, :tenure, :union, :wks_work, :msp])
    vce = ClusterVCE(data, :idcode, 6, 8)
    eq = (:ln_wage, (:tenure=>[:union, :wks_work, :msp], :age, :age2, :birth_yr, :grade))
    r = fit(IteratedLinearGMM, vce, data, eq, maxiter=2)

    # Stata ivregress Example 4
    # ivregress gmm ln_wage age age2 birth_yr grade (tenure = union wks_work msp),
    #    wmatrix(cluster idcode)
    @test coef(r) ≈ [0.09922101, 0.01711462, -0.0005191,
        -0.00859937, 0.07157395, 0.85750707] atol=1e-6
    @test stderror(r) ≈ [0.0037764, 0.0066895, 0.000111,
        0.0021932, 0.0029938, 0.1616274] atol=1e-6
end

@testset "JustIdentifiedLinearGMM" begin
    data = exampledata(:auto)
    vce = RobustVCE(3, 3, size(data,1))
    eq = (:mpg, [:weight, :length])
    r = fit(JustIdentifiedLinearGMM, vce, data, eq)
    # gmm (mpg - {b1}*weight - {b2}*length - {b0}), instruments(weight length) onestep
    @test coef(r) ≈ [-0.00385148, -0.07959347, 47.884873] atol=1e-6
    @test stderror(r) ≈ [0.0019472, 0.0677536, 7.506064] atol=1e-4

    @test sprint(show, MIME("text/plain"), r)[1:726] == """
        LinearGMM with 3 moments and 3 parameters:
          Just-identified linear GMM estimator
          Heteroskedasticity-robust covariance estimator
        ──────────────────────────────────────────────────────────────────────────
                   Estimate  Std. Error      z  Pr(>|z|)    Lower 95%    Upper 95%
        ──────────────────────────────────────────────────────────────────────────
        weight  -0.00385148  0.00194717  -1.98    0.0479  -0.00766786  -3.50918e-5"""

    eq = (:mpg, [:weight, :length=>:trunk])
    r = fit(JustIdentifiedLinearGMM, vce, data, eq)
    # gmm (mpg - {b1}*weight - {b2}*length - {b0}), instruments(weight trunk) onestep
    @test coef(r) ≈ [-0.00298026, -0.11173826, 51.295324] atol=1e-6
    @test stderror(r) ≈ [0.0045492, 0.1567287, 15.7791] atol=1e-4
end
