@testset "Domains" begin
    @info "Testing Domain primitives"

    @testset "hierarchy" begin
        @test Nominal <: Categorical
        @test Ordinal <: Categorical
        @test Categorical <: Domain
        @test Continuous <: Domain
        @test !(Continuous <: Categorical) # order there is continuous/enforced by construction, not part of this hierarchy
        @test !(Nominal <: Ordinal)
        @test !(Ordinal <: Nominal)
    end

    nom = Nominal(4)
    @test nom isa Domain
    @test nom isa Categorical
    @test nom.levels == 4

    ord = Ordinal(3)
    @test ord isa Domain
    @test ord isa Categorical
    @test ord.levels == 3

    c = Continuous(1, 5, 1) # dt=1 -> grid points 1,2,3,4,5
    @test c isa Domain
    @test !(c isa Categorical)
    @test c.min == 1.0
    @test c.max == 5.0
    @test c.dt == 1.0
    @test Hyperopt.nlevels(c) == 5

    # dt need not evenly divide (max-min): the grid stops at or before max, never past it.
    cshort = Continuous(0, 1, 0.3) # points at 0, 0.3, 0.6, 0.9 -- last point (0.9) falls short of 1.0
    @test Hyperopt.nlevels(cshort) == 4

    @test_throws ArgumentError Continuous(1, 5, 0)   # dt must be > 0
    @test_throws ArgumentError Continuous(1, 5, -1)
    @test_throws ArgumentError Continuous(5, 1, 1)   # max must be >= min

    @testset "Ordinal order checking" begin
        # Numeric values: order is verifiable, so it's actually enforced.
        o_sorted = Ordinal([1, 2, 5, 10])
        @test o_sorted.values == [1, 2, 5, 10]
        @test_throws ArgumentError Ordinal([1, 5, 2, 10])  # not sorted
        @test_throws ArgumentError Ordinal([3, 2, 1])

        # Non-numeric values: default `isless`/alphabetical order does not
        # reliably match the intended domain order (this is exactly why
        # "low" < "medium" < "high" isn't alphabetically sorted), so it can't
        # be verified -- construction still succeeds, preserving the given
        # order, but warns that it's assuming that order is intentional.
        local o_strings
        @test_logs (:warn, r"assuming this is the intended order") begin
            o_strings = Ordinal(["low", "medium", "high"])
        end
        @test o_strings.values == ["low", "medium", "high"] # given order preserved, NOT alphabetically resorted

        # A plain level count has no values to check order of -- 1:levels is
        # ordered by definition, nothing to warn about or reject.
        @test_logs min_level=Logging.Warn Ordinal(5)
    end

    @testset "rand, uniform (no weights)" begin
        rng = StableRNG(1)

        @test all(rand(rng, nom) in 1:4 for _ in 1:1000)
        @test all(rand(rng, ord) in 1:3 for _ in 1:1000)
        @test all(1:1000) do _
            v = rand(rng, c)
            1.0 <= v <= 5.0
        end
        # every Continuous draw must land exactly on the discretization grid
        @test all(1:1000) do _
            v = rand(rng, c)
            idx = round(Int, (v - c.min) / c.dt)
            isapprox(v, c.min + idx * c.dt; atol=1e-9)
        end

        @test rand(nom) isa Int   # no explicit rng also works
        @test rand(ord) isa Int
        @test rand(c) isa Float64
    end

    @testset "rand, weighted" begin
        rng = StableRNG(1)

        # Weight everything onto a single level/point -- draws must always land there.
        nom_w = Nominal(4; weights=[0.0, 1.0, 0.0, 0.0])
        @test all(rand(rng, nom_w) == 2 for _ in 1:200)

        ord_w = Ordinal(3; weights=[1.0, 0.0, 0.0])
        @test all(rand(rng, ord_w) == 1 for _ in 1:200)

        c_w = Continuous(1, 5, 1; weights=[0.0, 0.0, 0.0, 0.0, 1.0]) # forces the last grid point, 5.0
        @test all(rand(rng, c_w) == 5.0 for _ in 1:200)

        # Wrong-length weights are rejected at construction, not silently at sample time.
        @test_throws ArgumentError Nominal(4; weights=[1.0, 2.0])
        @test_throws ArgumentError Ordinal(3; weights=[1.0, 2.0, 3.0, 4.0])
        @test_throws ArgumentError Continuous(1, 5, 1; weights=[1.0, 2.0])
    end

    @testset "values-based construction (Domain replaces a plain candidate array)" begin
        rng = StableRNG(1)

        nom_v = Nominal([true, false]) # unordered -- Nominal never checks order
        @test nom_v.levels == 2
        @test all(rand(rng, nom_v) isa Bool for _ in 1:100)
        @test all(rand(rng, nom_v) in nom_v for _ in 1:100)

        nom_fns = Nominal([tanh, exp, identity]) # functions have no natural order -- Nominal, not Ordinal
        @test nom_fns.levels == 3
        @test all(rand(rng, nom_fns) in (tanh, exp, identity) for _ in 1:100)
        @test all(rand(rng, nom_fns) in nom_fns for _ in 1:100)

        # log-spaced (non-uniform, but genuinely increasing) values: only
        # representable via Ordinal's values form, not Continuous.
        logspaced = Ordinal(exp10.(LinRange(-1, 3, 50)))
        @test all(rand(rng, logspaced) in logspaced for _ in 1:200)

        # Domains constructed from a plain level count (no values) don't carry actual
        # candidate values -- membership only checks against the index range 1:levels.
        @test 1 in nom
        @test 5 ∉ nom
        @test "low" ∉ nom # non-integer -- never a member of an index-only domain

        # Continuous membership: on-grid values are members, off-grid values are not.
        @test 1.0 in c
        @test 3.0 in c
        @test 3.5 ∉ c   # c has dt=1, so 3.5 isn't on the grid
        @test 6.0 ∉ c   # out of [min,max]
    end
end
