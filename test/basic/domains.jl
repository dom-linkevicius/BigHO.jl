@testset "Domains" begin
    @info "Testing Domain primitives"

    nom = Nominal(4)
    ord = Ordinal(3)
    c = Continuous(1, 5)

    @testset "construction" begin
        @test nom isa BigHO.Domain
        @test ord isa BigHO.Domain
        @test c isa BigHO.Domain
        @test nom.values == 1:4
        @test ord.values == 1:3
        @test (c.min, c.max) == (1.0, 5.0)
        @test c.transform === identity

        # Explicit values, not just a level count.
        @test Nominal([tanh, exp]).values == [tanh, exp]
        @test Ordinal([1, 2, 5, 10]).values == [1, 2, 5, 10]

        # An empty domain would only fail later, at the first draw -- rejected at construction.
        @test_throws ArgumentError Nominal(0)
        @test_throws ArgumentError Ordinal(0)
        @test_throws ArgumentError Nominal(Int[])
        @test_throws ArgumentError Ordinal(Int[])

        # Continuous is an interval, so a degenerate or inverted one is rejected.
        @test_throws ArgumentError Continuous(5, 1)
        @test_throws ArgumentError Continuous(1, 1)
    end

    @testset "Ordinal order checking" begin
        # Numeric values: order is verifiable, so it's actually enforced.
        @test_throws ArgumentError Ordinal([1, 5, 2, 10])  # not sorted
        @test_throws ArgumentError Ordinal([3, 2, 1])

        # Non-numeric values: default `isless`/alphabetical order does not reliably match the
        # intended domain order (this is exactly why "low" < "medium" < "high" isn't alphabetically
        # sorted), so it can't be verified -- construction still succeeds, preserving the given
        # order, but warns that it's assuming that order is intentional.
        local o_strings
        @test_logs (:warn, r"assuming this is the intended order") begin
            o_strings = Ordinal(["low", "medium", "high"])
        end
        @test o_strings.values == ["low", "medium", "high"] # given order preserved, NOT alphabetically resorted

        # A plain level count has no values to check order of -- 1:levels is ordered by definition.
        @test_logs min_level = Logging.Warn Ordinal(5)
    end

    @testset "transform" begin
        log_d = Continuous(-4, -1; transform=exp10)
        @test BigHO.from_unit(log_d, 0.0) ≈ 1e-4
        @test BigHO.from_unit(log_d, 1.0) ≈ 1e-1
        @test BigHO.from_unit(log_d, 0.5) ≈ exp10(-2.5) # uniform in u is uniform in log space, not in the value

        # Nothing ever inverts the transform, so it needn't be monotonic.
        @test Continuous(0, 2π; transform=sin) isa BigHO.Domain

        # It does have to be finite at both ends, which is the only place it's checked.
        @test_throws ArgumentError Continuous(0, 1; transform=x -> 1 / x)
        @test_throws ArgumentError Continuous(0, Inf)
    end

    @testset "from_unit" begin
        levels = Nominal([:a, :b, :c, :d])

        # Equal-width bins over the whole interval: the endpoints land in the first/last bin.
        @test BigHO.from_unit(levels, 0.0) === :a
        @test BigHO.from_unit(levels, 1.0) === :d
        @test [BigHO.from_unit(levels, (i - 0.5) / 4) for i in 1:4] == [:a, :b, :c, :d]

        # A bin boundary belongs to the bin above it.
        @test BigHO.from_unit(levels, 0.25) === :a
        @test BigHO.from_unit(levels, nextfloat(0.25)) === :b

        @test BigHO.from_unit(c, 0.0) == 1.0
        @test BigHO.from_unit(c, 1.0) == 5.0
        @test BigHO.from_unit(c, 0.25) == 2.0

        # u outside [0,1] clamps rather than erroring or extrapolating past the domain -- a sampler
        # doing arithmetic in unit space can overshoot, and the result must still be a candidate.
        @test BigHO.from_unit(levels, -1.0) === :a
        @test BigHO.from_unit(levels, 2.0) === :d
        @test BigHO.from_unit(c, -1.0) == 1.0
        @test BigHO.from_unit(c, 3.0) == 5.0
    end

    @testset "rand" begin
        rng = StableRNG(1)

        @test all(rand(rng, nom) in 1:4 for _ in 1:1000)
        @test all(rand(rng, ord) in 1:3 for _ in 1:1000)
        @test all(1.0 <= rand(rng, c) <= 5.0 for _ in 1:1000)
        @test all(1e-4 <= rand(rng, Continuous(-4, -1; transform=exp10)) <= 1e-1 for _ in 1:200)

        # Every level is reachable, not just the interior ones.
        @test Set(rand(rng, nom) for _ in 1:1000) == Set(1:4)

        nom_fns = Nominal([tanh, exp, identity]) # functions have no natural order -- Nominal, not Ordinal
        @test all(rand(rng, nom_fns) in (tanh, exp, identity) for _ in 1:100)

        @test rand(nom) isa Int # the default rng also works
        @test rand(ord) isa Int
        @test rand(c) isa Float64
    end

    @testset "length" begin
        @test length(nom) == 4
        @test length(ord) == 3
        @test length(Nominal([tanh, exp])) == 2
        @test_throws MethodError length(c) # Continuous has no levels to count
    end
end
