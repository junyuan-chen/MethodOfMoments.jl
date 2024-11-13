@testset "BayesianGMM" begin
    data = Table(exampledata(:docvisits))
    g = g_stata_gmm_ex8(data)
    dg = dg_stata_gmm_ex8(data)
    vce = RobustVCE(5, 7, length(data))
    params = (:private=>Uniform(-1,2), :chronic=>Uniform(-1,2),
        :female=>Uniform(-1,2), :income=>Uniform(-1,2), :cons=>Normal())
    m = BayesianGMM(vce, g, dg, params, 7, length(data))
    θ = [0.5, 1, 1, 0.1, 0.1]

    @test dimension(m) == length(params)
    @test capabilities(m) == LogDensityOrder{1}()
    lpri = logprior(m, θ)
    lp = 0.0
    for (i, (n, d)) in enumerate(params)
        lp += logpdf(d, θ[i])
    end
    @test lpri ≈ lp
    l = logposterior!(m, θ)
    @test coef(m) == θ
    @test coef(m, :chronic) == θ[2]
    @test coefnames(m) == m.params
    @test parent(m) === m
    @test m[] == coef(m)
    @test m[2] == coef(m)[2]
    @test logdensity(m, θ) == l
    @test_throws ErrorException logposterior_and_gradient!(m, θ)

    @test sprint(show, m) == "7×5 BayesianGMM"
    @test sprint(show, MIME("text/plain"), m)[1:182] == """
        BayesianGMM with 7 moments and 5 parameters:
          private = 5.00000e-01  chronic = 1.00000e+00  female = 1.00000e+00  income = 1.00000e-01  cons = 1.00000e-01
          log(posterior) = -2.6803"""

    m1 = BayesianGMM(vce, g, dg, collect(params), 7, length(data); deriv=ForwardDiff, ntasks=2)
    l1, dl1 = logdensity_and_gradient(m1, θ)
    @test l1 ≈ l
    @test dl1 ≈ [-1.762191542650967, -1.0913702611945668, -0.980917818008333,
        -344.2501598611761, -2.100024393614612] atol=1e-4
    @test dl1 !== m.dl
    spl = MetropolisHastings(RandomWalkProposal{true}(MvNormal(zeros(5), 0.5*I(5))))
    Ndrop = 3000
    N = 6000
    @time chain = sample(m1, spl, N, init_params=θ,
        param_names=m1.params, chain_type=Chains, progress=false)
    ar = acceptance_rate(view(chain.value, Ndrop+1:N, 1, 1))
    @test 0.1 < ar < 0.9

    opt = NLopt.Opt(:LN_BOBYQA, length(params))
    r = mode(m1, opt, θ)
    @test r[3][1] > -5.3161

    data = exampledata(:poisson1)
    for v in (:mu, :dmu1, :dmu2, :dmu3)
        data[!,v] .= 0.0
    end
    mubar = zeros(45)
    dmubar = zeros(3,45)
    ybar = zeros(45)
    Nid = sort!(combine(groupby(data, :id), nrow=>:Nid), :id).Nid
    data = Table(data)
    params = (:x1, :x2, :x3).=>Uniform(-3,3)

    pg = preg_stata_gmm_ex11(data, mubar, ybar, Nid)
    g = g_stata_gmm_ex11(data, mubar, ybar)
    pdg = predg_stata_gmm_ex11(data, dmubar, ybar, Nid)
    dg = dg_stata_gmm_ex11(data, mubar, dmubar, ybar)
    vce = ClusterVCE(data, :id, 3, 3)

    θ = [2.0, -3.0, 1.0]
    m = BayesianGMM(vce, g, nothing, params, 3, length(data); preg=pg, ntasks=2)
    spl = MetropolisHastings(RandomWalkProposal{true}(MvNormal(zeros(3), I(3))))
    @time chain = sample(m, spl, N, init_params=θ,
        param_names=m.params, chain_type=Chains, progress=false)
end
