@testset "LHSampler" begin
    @info "Testing LHSampler"

    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

    # A mix of Continuous, Ordinal, and Nominal domains -- Continuous/Ordinal need
    # exactly n values (one design slot each); Nominal can have any number of levels.
    ho = Hyperoptimizer((a, b, c) -> f(a, b, c=c),
                         (a=Continuous(LinRange(1, 5, 100)),
                          b=Nominal([true, false]),
                          c=Ordinal(exp10.(LinRange(-1, 3, 100))));
                         sampler=LHSampler(), n=100)
    run!(ho)
    @test minimum(ho) < 300
    @test length(history(ho)) == 100
    @test length(results(ho)) == 100
    @test all(history(ho)) do h
        all(hi in ho.candidates[i] for (i, hi) in enumerate(h))
    end

    # Space-filling: a Continuous/Ordinal domain's n values are each used exactly
    # once across the n trials -- the defining property of a Latin Hypercube design.
    a_draws = [h[1] for h in history(ho)]
    @test sort(a_draws) == sort(collect(ho.candidates[1].values))

    # LHSampler is a FixedPlanSampler: requires ho.n up front, and can't be
    # resumed by raising the target -- the design is only valid for its original size.
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Continuous(1:10),); sampler=LHSampler())
    @test_throws ArgumentError settarget!(ho, 200)

    # A Continuous/Ordinal domain must have exactly n values -- the design assigns
    # each of its n slots exactly once, so a mismatched length can't be filled correctly.
    # init! runs inside the Hyperoptimizer constructor, so this throws immediately, not at run!.
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Continuous(1:10),); sampler=LHSampler(), n=100)

    # Nominal domains are exempt from the n-length requirement.
    ho_nominal = Hyperoptimizer((a, b) -> a + b, (a=Continuous(1:50), b=Nominal([1, 2, 3])); sampler=LHSampler(), n=50)
    run!(ho_nominal)
    @test length(results(ho_nominal)) == 50

    # weights aren't supported -- LHC is a deterministic space-filling design, not a weighted draw.
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Continuous(1:10; weights=fill(0.1, 10)),); sampler=LHSampler(), n=10)

    # gens is a pure optimization-quality knob -- any positive value must still produce a full, valid design.
    ho_gens = Hyperoptimizer(a -> a, (a=Continuous(1:20),); sampler=LHSampler(gens=5), n=20)
    run!(ho_gens)
    @test sort([h[1] for h in history(ho_gens)]) == sort(collect(ho_gens.candidates[1].values))
end
