@testset "LHSampler" begin
    @info "Testing LHSampler"

    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

    # A mix of Continuous, Ordinal, and Nominal domains, all already matching n=100 --
    # no override warning expected here (see the dedicated test for that below).
    # Continuous(min,max,dt) is used throughout this file, not Continuous(values) --
    # LHSampler rejects the latter (arbitrary spacing), see the dedicated test below.
    ho = Hyperoptimizer((a, b, c) -> f(a, b, c=c),
                         (a=Continuous(1, 5, 4 / 99),
                          b=Nominal([true, false]),
                          c=Ordinal(exp10.(LinRange(-1, 3, 100)))),
                         LHSampler(); n=100)
    run!(ho)
    @test minimum(ho) < 300
    @test length(history(ho)) == 100
    @test length(results(ho)) == 100
    @test all(history(ho)) do h
        all(hi in ho.candidates[i] for (i, hi) in enumerate(h))
    end

    # Space-filling: a Continuous domain's n values are each used exactly once
    # across the n trials -- the defining property of a Latin Hypercube design.
    a_draws = [h[1] for h in history(ho)]
    @test sort(a_draws) == sort(collect(ho.candidates[1].values))

    # LHSampler is a FixedPlanSampler: can't be resumed by raising the target --
    # the design is only valid for its original size.
    @test_throws ArgumentError settarget!(ho, 200)

    # n is required for a FixedPlanSampler via the plain keyword constructor --
    # use Hyperoptimizer(objective, candidates, sampler; n=...) instead.
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Continuous(1, 10, 1),); sampler=LHSampler())

    # Via the plain keyword constructor (no rebuild), a Continuous domain must
    # still have exactly n values -- only the dedicated constructor rebuilds it.
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Continuous(1, 10, 1),); sampler=LHSampler(), n=100)

    # Continuous(values) (arbitrary spacing) is rejected outright, via either constructor --
    # LHSampler always assumes linear spacing; apply nonlinear transforms in the objective.
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Continuous(exp10.(LinRange(-1, 3, 50))),); sampler=LHSampler(), n=50)
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Continuous(exp10.(LinRange(-1, 3, 50))),), LHSampler(); n=50)

    # The dedicated constructor rebuilds a mismatched Continuous(min,max,dt) domain
    # instead of erroring -- see "LHSampler overrides Continuous domains" for the warning.
    ho_rebuilt = Hyperoptimizer(a -> a, (a=Continuous(1, 10, 1),), LHSampler(); n=100)
    @test length(ho_rebuilt.candidates[1].values) == 100

    # Nominal/Ordinal domains are exempt from the n-length requirement -- a small,
    # fixed set of named levels (e.g. "low"/"medium"/"high") over many more trials.
    ho_nominal = Hyperoptimizer((a, b) -> a + b, (a=Continuous(1, 50, 1), b=Nominal([1, 2, 3])); sampler=LHSampler(), n=50)
    run!(ho_nominal)
    @test length(results(ho_nominal)) == 50

    ho_ordinal = Hyperoptimizer((a, b) -> a + length(b), (a=Continuous(1, 50, 1), b=Ordinal(["low", "medium", "high"])); sampler=LHSampler(), n=50)
    run!(ho_ordinal)
    @test length(results(ho_ordinal)) == 50

    # weights aren't supported -- LHC is a deterministic space-filling design, not a weighted draw.
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Continuous(1, 10, 1; weights=fill(0.1, 10)),); sampler=LHSampler(), n=10)

    # gens is a pure optimization-quality knob -- any positive value must still produce a full, valid design.
    ho_gens = Hyperoptimizer(a -> a, (a=Continuous(1, 20, 1),); sampler=LHSampler(gens=5), n=20)
    run!(ho_gens)
    @test sort([h[1] for h in history(ho_gens)]) == sort(collect(ho_gens.candidates[1].values))
end

@testset "LHSampler overrides Continuous domains" begin
    @info "Testing LHSampler overrides Continuous domains"

    # The dedicated constructor rebuilds a Continuous(min,max,dt) domain to exactly n
    # linearly-spaced values, warning about it -- deterministic (unlike the GA's own outcome).
    @test_logs (:warn, r"overriding `a`.*10 linearly-spaced values") match_mode = :any begin
        ho = Hyperoptimizer(a -> a, (a=Continuous(1, 5, 1),), LHSampler(); n=10)
        @test length(ho.candidates[1].values) == 10
        @test collect(ho.candidates[1].values) == collect(LinRange(1, 5, 10))
    end

    # No warning when the domain already matches n -- nothing is actually being overridden.
    @test_logs min_level = Logging.Warn Hyperoptimizer(a -> a, (a=Continuous(1, 10, 1),), LHSampler(); n=10)
end

@testset "LHSampler discrete-combination warnings" begin
    @info "Testing LHSampler discrete-combination warnings"

    # n < the full product of discrete levels: deterministic (doesn't depend on the
    # GA's randomized outcome), so safe to check via an actual construction.
    @test_logs (:warn, r"n \(5\) is less than the number of discrete-variable combinations \(9\)") match_mode = :any begin
        Hyperoptimizer((a, b) -> 0.0, (a=Nominal(["x", "y", "z"]), b=Nominal([1, 2, 3])); sampler=LHSampler(), n=5)
    end

    # The "design doesn't cover every combination" check depends on the GA's own
    # (randomized, unseeded) outcome, which isn't guaranteed to leave a gap -- so it's
    # exercised directly against a synthetic design instead of a real, flaky run.
    # Deliberately never uses level combination (2, 2).
    candidates = values((a=Nominal(["x", "y"]), b=Nominal(["p", "q"])))
    design = [1 1; 1 2; 2 1; 1 1]
    @test_logs (:warn, r"doesn't cover every discrete-variable combination") BigHO._warn_missing_combinations(design, candidates)

    # A design that does cover everything logs nothing.
    full_design = [1 1; 1 2; 2 1; 2 2]
    @test_logs min_level = Logging.Warn BigHO._warn_missing_combinations(full_design, candidates)
end

@testset "LHSampler optimization correctness" begin
    @info "Testing LHSampler optimization correctness"

    # 1D: LHC visits every one of a single Continuous domain's n values exactly
    # once -- for a single dimension this IS an exhaustive sweep, so the true
    # global optimum is found deterministically, not just with high probability.
    ho1 = Hyperoptimizer(x -> (x - 37)^2, (x=Continuous(1, 100, 1),), LHSampler(gens=50); n=100)
    run!(ho1)
    @test minimum(ho1) == 0
    @test minimizer(ho1) == [37]

    # 2D: LHC only guarantees per-dimension coverage, not joint combinatorial
    # coverage, so the true optimum isn't necessarily hit exactly -- but its
    # deliberate spread still converges much closer than n=100 random draws
    # would over the same 100x100 grid.
    target = (33.0, 68.0)
    ho2 = Hyperoptimizer((x, y) -> (x - target[1])^2 + (y - target[2])^2,
                         (x=Continuous(1, 100, 1), y=Continuous(1, 100, 1)),
                         LHSampler(gens=50); n=100)
    run!(ho2)
    m = minimizer(ho2)
    @test abs(m[1] - target[1]) < 10
    @test abs(m[2] - target[2]) < 10
end
