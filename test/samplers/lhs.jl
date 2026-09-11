@testset "LHSampler" begin
    @info "Testing LHSampler"

    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

    # The defining Latin-hypercube property, checked in unit space: every dimension proposes each
    # of the n equal-width strata exactly once, so this is `1:n` for any dimension of any design.
    strata_hit(ho, dim) = sort(ceil.(Int, [e.unit_params[dim] for e in ho.runs] .* length(ho.runs)))

    # A mix of Continuous, Ordinal, and Nominal domains. Ordinal stays a small, realistic level
    # count -- not a near-continuous sweep, which is what Continuous is for.
    ho = Hyperoptimizer(p -> f(p.a, p.b, c=p.c),
                         (a=Continuous(1, 5),
                          b=Nominal([true, false]),
                          c=Ordinal([1, 10, 100, 1000])),
                         LHSampler(); n=100)
    run!(ho)
    @test minimum(ho) < 300
    @test length(history(ho)) == 100
    @test length(results(ho)) == 100
    @test all(h -> 1.0 <= h.a <= 5.0 && h.b isa Bool && h.c in (1, 10, 100, 1000), history(ho))

    @test !BigHO.blocked(ho.sampler, ho)

    # Space-filling, in every dimension -- discrete ones included, since they're stratified over n
    # too and only collapse onto levels at decode.
    @test all(dim -> strata_hit(ho, dim) == 1:100, 1:3)

    # LHSampler is a FixedPlanSampler: can't be resumed by raising the target --
    # the design is only valid for its original size.
    @test_throws ArgumentError settarget!(ho, 200)

    # n is required for a FixedPlanSampler via the plain keyword constructor --
    # use Hyperoptimizer(objective, candidates, sampler; n=...) instead.
    @test_throws ArgumentError Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),); sampler=LHSampler())

    # An already-initialized sampler is rejected rather than silently re-drawn for the new n.
    @test_throws ArgumentError Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),), ho.sampler; n=50)
    # An untouched one can still seed as many optimizers as you like -- init never mutates it.
    fresh = LHSampler()
    @test Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),), fresh; n=5) isa Hyperoptimizer
    @test Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),), fresh; n=9) isa Hyperoptimizer

    # A Continuous domain carries no level count of its own, so any n works with any domain -- the
    # strata come from n, and the domain only decodes them.
    ho_any_n = Hyperoptimizer(p -> p.a, (a=Continuous(1, 10),); sampler=LHSampler(), n=7)
    run!(ho_any_n)
    @test strata_hit(ho_any_n, 1) == 1:7

    # A non-linear domain works the same way -- the design is over strata, and the transform is
    # applied on decode, so log-spaced candidates need no special handling.
    ho_log = Hyperoptimizer(p -> p.a, (a=Continuous(-1, 3; transform=exp10),); sampler=LHSampler(), n=20)
    run!(ho_log)
    @test all(h -> 0.1 <= h.a <= 1000.0, history(ho_log))
    @test strata_hit(ho_log, 1) == 1:20

    # Nominal/Ordinal domains are stratified over their own levels, independently of n -- a small,
    # fixed set of named levels (e.g. "low"/"medium"/"high") over many more trials.
    ho_nominal = Hyperoptimizer(p -> p.a + p.b, (a=Continuous(1, 50), b=Nominal([1, 2, 3])); sampler=LHSampler(), n=50)
    run!(ho_nominal)
    @test length(results(ho_nominal)) == 50

    ho_ordinal = Hyperoptimizer(p -> p.a + length(p.b), (a=Continuous(1, 50), b=Ordinal(["low", "medium", "high"])); sampler=LHSampler(), n=50)
    run!(ho_ordinal)
    @test length(results(ho_ordinal)) == 50

    # The design is drawn from the sampler's own rng, so the same seed gives the same design and a
    # different one doesn't -- previously it came from the global RNG and wasn't reproducible at all.
    plan(rng) = (ho = Hyperoptimizer(p -> p.a, (a=Continuous(1, 20),); sampler=LHSampler(; rng), n=20);
                 run!(ho); [h.a for h in history(ho)])
    @test plan(StableRNG(7)) == plan(StableRNG(7))
    @test plan(StableRNG(7)) != plan(StableRNG(8))
end

@testset "LHSampler discrete-combination warnings" begin
    @info "Testing LHSampler discrete-combination warnings"

    # n < the full product of discrete levels: known before any design is drawn.
    @test_logs (:warn, r"n \(5\) is less than the number of discrete-variable combinations \(9\)") match_mode = :any begin
        Hyperoptimizer(p -> 0.0, (a=Nominal(["x", "y", "z"]), b=Nominal([1, 2, 3])); sampler=LHSampler(), n=5)
    end

    # Whether a real design leaves a gap depends on the draw, so this is exercised against a
    # synthetic one: stratum centres for n=4, deliberately never pairing ("y", "q").
    candidates = values((a=Nominal(["x", "y"]), b=Nominal(["p", "q"])))
    design = [0.125 0.125 0.625 0.375
              0.125 0.625 0.125 0.375]
    @test_logs (:warn, r"doesn't cover every discrete-variable combination") BigHO._warn_missing_combinations(design, candidates)

    # A design that does cover everything logs nothing.
    full_design = [0.125 0.375 0.625 0.875
                   0.125 0.625 0.375 0.875]
    @test_logs min_level = Logging.Warn BigHO._warn_missing_combinations(full_design, candidates)
end

@testset "LHSampler optimization correctness" begin
    @info "Testing LHSampler optimization correctness"

    # 1D: LHC visits every one of n strata exactly once -- for a single dimension that IS an
    # exhaustive sweep at stratum resolution, so the optimum is found to within half a stratum
    # deterministically, not just with high probability. 100 strata over [1, 100] -> ~0.5 wide.
    ho1 = Hyperoptimizer(p -> (p.x - 37)^2, (x=Continuous(1, 100),), LHSampler(); n=100)
    run!(ho1)
    @test abs(minimizer(ho1)[1] - 37) < 0.5

    # 2D: LHC only guarantees per-dimension coverage, not joint combinatorial
    # coverage, so the optimum isn't approached nearly as closely -- but its
    # deliberate spread still converges much closer than n=100 random draws would.
    target = (33.0, 68.0)
    ho2 = Hyperoptimizer(p -> (p.x - target[1])^2 + (p.y - target[2])^2,
                         (x=Continuous(1, 100), y=Continuous(1, 100)),
                         LHSampler(); n=100)
    run!(ho2)
    m = minimizer(ho2)
    @test abs(m[1] - target[1]) < 10
    @test abs(m[2] - target[2]) < 10
end
