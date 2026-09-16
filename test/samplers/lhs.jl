@testset "LHSampler" begin
    @info "Testing LHSampler"

    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

    strata_hit(ho, dim) = sort(ceil.(Int, [e.unit_params[dim] for e in ho.runs] .* length(ho.runs)))

    ho = Hyperoptimizer(p -> f(p.a, p.b, c=p.c),
                         (a=Continuous(1, 5),
                          b=Nominal([true, false]),
                          c=Ordinal([1, 10, 100, 1000])),
                         sampler=LHSampler(), n=100)
    run!(ho)
    @test minimum(ho) < 300
    @test length(history(ho)) == 100
    @test length(results(ho)) == 100
    @test all(h -> 1.0 <= h.a <= 5.0 && h.b isa Bool && h.c in (1, 10, 100, 1000), history(ho))

    @test !BigHO.blocked(ho.sampler, ho)

    @test all(dim -> strata_hit(ho, dim) == 1:100, 1:3)

    @test_throws ArgumentError settarget!(ho, 200)

    @test_throws ArgumentError Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),); sampler=ho.sampler, n=50)
    fresh = LHSampler()
    @test Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),); sampler=fresh, n=5) isa Hyperoptimizer
    @test Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),); sampler=fresh, n=9) isa Hyperoptimizer

    ho_any_n = Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),); sampler=LHSampler(), n=7)
    run!(ho_any_n)
    @test strata_hit(ho_any_n, 1) == 1:7

    ho_log = Hyperoptimizer(p -> p.a, (a=Continuous(-1, 3; transform=exp10),); sampler=LHSampler(), n=20)
    run!(ho_log)
    @test all(h -> 0.1 <= h.a <= 1000.0, history(ho_log))
    @test strata_hit(ho_log, 1) == 1:20

    ho_nominal = Hyperoptimizer(p -> p.a + p.b, (a=Continuous(1, 50), b=Nominal([1, 2, 3])); sampler=LHSampler(), n=50)
    run!(ho_nominal)
    @test length(results(ho_nominal)) == 50

    ho_ordinal = Hyperoptimizer(p -> p.a + length(p.b), (a=Continuous(1, 50), b=Ordinal(["low", "medium", "high"])); sampler=LHSampler(), n=50)
    run!(ho_ordinal)
    @test length(results(ho_ordinal)) == 50

    plan(rng) = (ho = Hyperoptimizer(p -> p.a, (a=Continuous(1, 20),); sampler=LHSampler(; rng), n=20);
                 run!(ho); [h.a for h in history(ho)])
    @test plan(StableRNG(7)) == plan(StableRNG(7))
    @test plan(StableRNG(7)) != plan(StableRNG(8))
end

@testset "LHSampler discrete-combination warnings" begin
    @info "Testing LHSampler discrete-combination warnings"

    @test_logs (:warn, r"n \(5\) is less than the number of discrete-variable combinations \(9\)") match_mode = :any begin
        Hyperoptimizer(p -> 0.0, (a=Nominal(["x", "y", "z"]), b=Nominal([1, 2, 3])); sampler=LHSampler(), n=5)
    end

    candidates = values((a=Nominal(["x", "y"]), b=Nominal(["p", "q"])))
    design = [0.125 0.125 0.625 0.375
              0.125 0.625 0.125 0.375]
    @test_logs (:warn, r"doesn't cover every discrete-variable combination") BigHO._warn_missing_combinations(design, candidates)

    full_design = [0.125 0.375 0.625 0.875
                   0.125 0.625 0.375 0.875]
    @test_logs min_level = Logging.Warn BigHO._warn_missing_combinations(full_design, candidates)
end

@testset "LHSampler optimization correctness" begin
    @info "Testing LHSampler optimization correctness"

    ho1 = Hyperoptimizer(p -> (p.x - 37)^2, (x=Continuous(1, 100),); sampler=LHSampler(), n=100)
    run!(ho1)
    @test abs(minimizer(ho1)[1] - 37) < 0.5

    target = (33.0, 68.0)
    ho2 = Hyperoptimizer(p -> (p.x - target[1])^2 + (p.y - target[2])^2,
                         (x=Continuous(1, 100), y=Continuous(1, 100));
                         sampler=LHSampler(), n=100)
    run!(ho2)
    m = minimizer(ho2)
    @test abs(m[1] - target[1]) < 10
    @test abs(m[2] - target[2]) < 10
end
