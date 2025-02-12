@gdg function (g::g_stata_iv_ex4)(θ, r)
    @inbounds d = g.data[r]
    x = SVector{6,Float64}((d.tenure, d.age, d.age2, d.birth_yr, d.grade, 1.0))
    z = SVector{8,Float64}((d.union, d.wks_work, d.msp, d.age, d.age2, d.birth_yr, d.grade, 1.0))
    return (d.ln_wage - θ'x) .* z
end

@gdg function (dg::dg_stata_iv_ex4)(θ, r)
    @inbounds d = dg.data[r]
    x = SVector{6,Float64}((d.tenure, d.age, d.age2, d.birth_yr, d.grade, 1.0))
    z = SVector{8,Float64}((d.union, d.wks_work, d.msp, d.age, d.age2, d.birth_yr, d.grade, 1.0))
    return z .* x'
end

@testset "CUGMM LinearCUGMM" begin
    data = exampledata(:nlswork)
    data[!,:age2] = data.age.^2
    dropmissing!(data, [:ln_wage, :age, :birth_yr, :grade, :tenure, :union, :wks_work, :msp])
    # This avoids type inference for SVector
    for col in [:age, :age2, :birth_yr, :grade, :union, :wks_work, :msp]
        data[!,col] = convert(Vector{Float64}, data[!,col])
    end
    data = Table(data)
    params = (:tenure, :age, :age2, :birth_yr, :grade, :cons)
    g = g_stata_iv_ex4(data)
    dg = dg_stata_iv_ex4(data)
    vce = RobustVCE(6, 8, length(data))
    @time r = fit(CUGMM, Hybrid, vce, g, dg, params, 8, length(data), ntasks=1;
        solverkwargs=(thres_jac=0,))
    # Compare results with Stata
    # ivreg2 ln_wage age c.age#c.age birth_yr grade (tenure = union wks_work msp), cue r
    b = [0.10747574, 0.01730719, -0.00055437, -0.00917305, 0.07049663, 0.89528047]
    se = [0.0033784, 0.0054639, 0.0000917, 0.0012685, 0.001708, 0.1030211]
    @test coef(r) ≈ b atol=1e-5
    @test stderror(r) ≈ se atol=1e-5

    # Initial value based on 2SLS
    p0 = (tenure=0.1060831918371669, age=0.016234501489638096, age2=-0.000530861756005715,
        birth_yr=-0.009113862811874278, grade=0.07045401220548421, cons=0.907953721661843)

    @time r0 = fit(CUGMM, Hybrid, vce, g, dg, p0, 8, length(data), ntasks=1)
    @test coef(r0) ≈ b atol=1e-5
    @test stderror(r0) ≈ se atol=1e-5

    @time r1 = fit(CUGMM, Hybrid, vce, g, dg, p0, 8, length(data), ntasks=2)
    @test coef(r1) ≈ coef(r0) atol=1e-6

    opt = NLopt.Opt(:LN_NELDERMEAD, length(params))
    opt.maxeval = 5000
    @time r2 = fit(CUGMM, opt, vce, g, dg, params, 8, length(data))
    @test coef(r2) ≈ b atol=1e-5
    @test stderror(r2) ≈ se atol=1e-5

    eq = (:ln_wage, (:tenure=>[:union, :wks_work, :msp], :age, :age2, :birth_yr, :grade))
    @time r3 = fit(LinearCUGMM, Hybrid, vce, data, eq)
    @test coef(r3) ≈ b atol=5e-4
    @test stderror(r3) ≈ se atol=1e-5

    opt = NLopt.Opt(:LN_NELDERMEAD, length(params))
    opt.maxeval = 5000
    @time r4 = fit(LinearCUGMM, opt, vce, data, eq)
    @test coef(r4) ≈ b atol=1e-6
    @test stderror(r4) ≈ se atol=1e-6

    @time r5 = fit(LinearCUGMM, Hybrid, vce, data, eq; θ0=zeros(6))
    @test coef(r5) ≈ b atol=1e-2
    @test stderror(r5) ≈ se atol=1e-4

    str = """
        NonlinearGMM with 8 moments and 6 parameters over 18625 observations:
          Continuously updated GMM estimator:
            Jstat = 31.47        Pr(>J) = <1e-06
          Heteroskedasticity-robust covariance estimator"""
    @test sprint(show, r.est) == "CUGMM"
    @test sprint(show, MIME("text/plain"), r)[1:197] == str
    @test sprint(show, r3.est) == "LinearCUGMM"
    @test sprint(show, MIME("text/plain"), r3)[1:197] == str

    vce = ClusterVCE(data, :idcode, 6, 8)
    @time r = fit(CUGMM, Hybrid, vce, g, dg, params, 8, length(data))
    # Compare results with Stata
    # ivreg2 ln_wage age c.age#c.age birth_yr grade (tenure = union wks_work msp), cue cluster(idcode)
    b = [0.10018515, 0.01695964, -0.00051943, -0.00867608, 0.0714917, 0.86427816]
    se = [0.0037927, 0.0067238, 0.0001116, 0.0022018, 0.0030075, 0.1624299]
    @test coef(r) ≈ b atol=1e-3
    @test stderror(r) ≈ se atol=1e-4
    @test Jstat(r) ≈ 12.685 atol=1e-3

    opt = NLopt.Opt(:LN_NELDERMEAD, length(params))
    @time r2 = fit(LinearCUGMM, opt, vce, data, eq; solverkwargs=(maxeval=5000,))
    @test coef(r2) ≈ b atol=1e-6
    @test stderror(r2) ≈ se atol=1e-6

    opt = NLopt.Opt(:LN_BOBYQA, length(params))
    @time r3 = fit(LinearCUGMM, opt, vce, data, eq)
    @test coef(r3) ≈ b atol=1e-4
    @test stderror(r3) ≈ se atol=1e-6

    eq = (:ln_wage, (:tenure, :age), (:union,))
    @test_throws ArgumentError fit(LinearCUGMM, Hybrid, vce, data, eq)
end
