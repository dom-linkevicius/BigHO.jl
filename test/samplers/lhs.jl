@testset "LHSampler" begin
    @info "Testing LHSampler"

    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

    # A design column holds one stratum index per trial, so a Continuous dimension's n strata are
    # each visited exactly once; a discrete one's strata are its own levels. Every value therefore
    # comes back as its stratum's centre in [0,1], decoded through the domain.
    stratum_centres(d::BigHO.Continuous, n) = [BigHO.from_unit(d, (i - 0.5) / n) for i in 1:n]

    # A mix of Continuous, Ordinal, and Nominal domains. Ordinal stays a small, realistic level
    # count -- not a near-continuous sweep, which is what Continuous is for.
    ho = Hyperoptimizer(p -> f(p.a, p.b, c=p.c),
                         (a=Continuous(1, 5),
                          b=Nominal([true, false]),
                          c=Ordinal([1, 10, 100, 1000])),
                         LHSampler(gens=100); n=100)
    run!(ho)
    @test minimum(ho) < 300
    @test length(history(ho)) == 100
    @test length(results(ho)) == 100
    @test all(h -> 1.0 <= h.a <= 5.0 && h.b isa Bool && h.c in (1, 10, 100, 1000), history(ho))

    @test !BigHO.blocked(ho.sampler, ho)

    # Space-filling: a Continuous dimension's n strata are each used exactly once across the n
    # trials -- the defining property of a Latin Hypercube design.
    @test sort([h.a for h in history(ho)]) == stratum_centres(ho.candidates[1], 100)

    # LHSampler is a FixedPlanSampler: can't be resumed by raising the target --
    # the design is only valid for its original size.
    @test_throws ArgumentError settarget!(ho, 200)

    # n is required for a FixedPlanSampler via the plain keyword constructor --
    # use Hyperoptimizer(objective, candidates, sampler; n=...) instead.
    @test_throws ArgumentError Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),); sampler=LHSampler(gens=10))

    # A Continuous domain carries no level count of its own, so any n works with any domain -- the
    # strata come from n, and the domain only decodes them.
    ho_any_n = Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),); sampler=LHSampler(gens=10), n=7)
    run!(ho_any_n)
    @test sort([h.a for h in history(ho_any_n)]) == stratum_centres(ho_any_n.candidates[1], 7)

    # A non-linear domain works the same way -- the design is over strata, and the transform is
    # applied on decode, so log-spaced candidates need no special handling.
    ho_log = Hyperoptimizer(p -> p.a, (a=Continuous(-1, 3; transform=exp10),); sampler=LHSampler(gens=10), n=20)
    run!(ho_log)
    @test all(h -> 0.1 <= h.a <= 1000.0, history(ho_log))
    @test sort([h.a for h in history(ho_log)]) == stratum_centres(ho_log.candidates[1], 20)

    # Nominal/Ordinal domains are stratified over their own levels, independently of n -- a small,
    # fixed set of named levels (e.g. "low"/"medium"/"high") over many more trials.
    ho_nominal = Hyperoptimizer(p -> p.a + p.b, (a=Continuous(1, 50), b=Nominal([1, 2, 3])); sampler=LHSampler(gens=50), n=50)
    run!(ho_nominal)
    @test length(results(ho_nominal)) == 50

    ho_ordinal = Hyperoptimizer(p -> p.a + length(p.b), (a=Continuous(1, 50), b=Ordinal(["low", "medium", "high"])); sampler=LHSampler(gens=50), n=50)
    run!(ho_ordinal)
    @test length(results(ho_ordinal)) == 50

    # gens is a pure optimization-quality knob -- any positive value must still produce a full, valid design.
    ho_gens = Hyperoptimizer(p -> p.a, (a=Continuous(1, 20),); sampler=LHSampler(gens=5), n=20)
    run!(ho_gens)
    @test sort([h.a for h in history(ho_gens)]) == stratum_centres(ho_gens.candidates[1], 20)

    # get_lhs_optim_history exposes the per-generation best fitness, one more than gens
    # (generation 0's initial best, then one per generation), non-decreasing since the
    # GA always carries its best design forward.
    hist = get_lhs_optim_history(ho_gens)
    @test hist isa Vector{Float64}
    @test length(hist) == 6
    @test issorted(hist)

    # Only defined for a Hyperoptimizer actually using LHSampler.
    ho_random = Hyperoptimizer(p -> p.a, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws ArgumentError get_lhs_optim_history(ho_random)
end

@testset "LHSampler discrete-combination warnings" begin
    @info "Testing LHSampler discrete-combination warnings"

    # n < the full product of discrete levels: deterministic (doesn't depend on the
    # GA's randomized outcome), so safe to check via an actual construction.
    @test_logs (:warn, r"n \(5\) is less than the number of discrete-variable combinations \(9\)") match_mode = :any begin
        Hyperoptimizer(p -> 0.0, (a=Nominal(["x", "y", "z"]), b=Nominal([1, 2, 3])); sampler=LHSampler(gens=20), n=5)
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

    # 1D: LHC visits every one of n strata exactly once -- for a single dimension that IS an
    # exhaustive sweep at stratum resolution, so the optimum is found to within half a stratum
    # deterministically, not just with high probability. 100 strata over [1, 100] -> ~0.5 wide.
    ho1 = Hyperoptimizer(p -> (p.x - 37)^2, (x=Continuous(1, 100),), LHSampler(gens=50); n=100)
    run!(ho1)
    @test abs(minimizer(ho1)[1] - 37) < 0.5

    # 2D: LHC only guarantees per-dimension coverage, not joint combinatorial
    # coverage, so the optimum isn't approached nearly as closely -- but its
    # deliberate spread still converges much closer than n=100 random draws would.
    target = (33.0, 68.0)
    ho2 = Hyperoptimizer(p -> (p.x - target[1])^2 + (p.y - target[2])^2,
                         (x=Continuous(1, 100), y=Continuous(1, 100)),
                         LHSampler(gens=50); n=100)
    run!(ho2)
    m = minimizer(ho2)
    @test abs(m[1] - target[1]) < 10
    @test abs(m[2] - target[2]) < 10
end
