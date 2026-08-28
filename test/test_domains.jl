@testset "Domains" begin
    @info "Testing Domain primitives"

    @testset "kind" begin
        @test Nominal(4) isa Hyperopt.Domain
        @test Ordinal(3) isa Hyperopt.Domain
        @test Continuous(1, 5, 1) isa Hyperopt.Domain
        @test Nominal(4).type == :nominal
        @test Ordinal(3).type == :ordinal
        @test Continuous(1, 5, 1).type == :continuous
    end

    nom = Nominal(4)
    @test nom isa Hyperopt.Domain
    @test length(nom.values) == 4

    ord = Ordinal(3)
    @test ord isa Hyperopt.Domain
    @test length(ord.values) == 3

    c = Continuous(1, 5, 1) # dt=1 -> grid points 1,2,3,4,5
    @test c isa Hyperopt.Domain
    @test first(c.values) == 1.0
    @test last(c.values) == 5.0
    @test c.values[2] - c.values[1] == 1.0
    @test length(c.values) == 5

    # dt need not evenly divide (max-min): the grid stops at or before max, never past it.
    cshort = Continuous(0, 1, 0.3) # points at 0, 0.3, 0.6, 0.9 -- last point (0.9) falls short of 1.0
    @test length(cshort.values) == 4

    @test_throws ArgumentError Continuous(1, 5, 0)   # dt must be > 0
    @test_throws ArgumentError Continuous(1, 5, -1)
    @test_throws ArgumentError Continuous(5, 1, 1)   # max must be >= min

    @testset "Continuous from values" begin
        # Equivalent to Continuous(1, 5, 1) -- built from an explicit,
        # increasing sequence instead of (min, max, dt). == compares by
        # value, not concrete container type, so a Vector-backed and a
        # range-backed Continuous with the same points compare equal.
        c_vals = Continuous(1:1:5)
        @test c_vals isa Hyperopt.Domain
        @test c_vals.type == :continuous
        @test c_vals.values == c.values

        # A plain, evenly-spaced Vector works too, not just a range.
        c_vec = Continuous([1.0, 1.5, 2.0, 2.5, 3.0])
        @test first(c_vec.values) == 1.0
        @test last(c_vec.values) == 3.0
        @test length(c_vec.values) == 5

        # Continuous does NOT require even spacing -- unlike the (min, max,
        # dt) form (which always produces a uniform grid by construction),
        # the values form accepts any increasing sequence, e.g. log-spaced.
        # What distinguishes Continuous from Ordinal is the :continuous tag
        # itself (relevant to future density-estimation machinery), not grid
        # uniformity.
        logspaced = Continuous(exp10.(LinRange(-1, 3, 50)))
        @test logspaced isa Hyperopt.Domain
        @test logspaced.type == :continuous
        @test length(logspaced.values) == 50
        @test all(rand(StableRNG(1), logspaced) in logspaced for _ in 1:200)
        @test logspaced.values[1] in logspaced
        @test logspaced.values[end] in logspaced
        @test (first(logspaced.values) - 1) ∉ logspaced # below the range
        @test (last(logspaced.values) + 1) ∉ logspaced  # above the range
        @test 1.0 ∉ logspaced # inside [min,max] but not one of the 50 log-spaced points

        # Duplicate values are rejected -- issorted alone allows equal
        # neighbors (issorted([1, 1, 2, 3]) is true), so allunique is
        # checked too, requiring strictly increasing values.
        @test_throws ArgumentError Continuous([1.0, 1.0, 2.0])

        # A single value is a valid (degenerate, 1-point) Continuous, same as
        # Continuous(5, 5, 1) already is via the (min, max, dt) form.
        @test length(Continuous([5.0]).values) == 1

        # Descending values are rejected, same as dt <= 0 is for the
        # (min, max, dt) form.
        @test_throws ArgumentError Continuous([5.0, 4.0, 3.0])

        # weights works identically to the (min, max, dt) form.
        c_vals_w = Continuous(1:1:5; weights=[0.0, 0.0, 0.0, 0.0, 1.0])
        rng = StableRNG(1)
        @test all(rand(rng, c_vals_w) == 5.0 for _ in 1:200)
        @test_throws ArgumentError Continuous(1:1:5; weights=[1.0, 2.0])
    end

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
        @test all(rand(rng, c) in c.values for _ in 1:1000)

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

        # All-zero weights are rejected too -- StatsBase.sample doesn't
        # validate this itself, so without this check it would silently
        # degrade into always drawing the same (first) candidate forever,
        # rather than throwing at construction like every other invalid
        # weights vector does.
        @test_throws ArgumentError Nominal(4; weights=[0.0, 0.0, 0.0, 0.0])
        # Negative weights are rejected too -- StatsBase.Weights has no
        # well-defined meaning for them.
        @test_throws ArgumentError Nominal(4; weights=[-1.0, 2.0, 0.0, 0.0])

        # Also rejected when constructing a Domain directly (bypassing
        # Nominal/Ordinal/Continuous) -- the check lives on Domain's own
        # inner constructor precisely so no construction path can skip it.
        @test_throws ArgumentError Hyperopt.Domain(:nominal, [1, 2, 3], [1.0])
        @test_throws ArgumentError Hyperopt.Domain(:bogus, [1, 2, 3], nothing)

        # Continuous's grid point count is derived from `length(min:dt:max)`,
        # not a hand-rolled floor((max-min)/dt)+1 formula -- Julia's range
        # length correctly counts 4 points here (0.0, 0.1, 0.2, 0.3) even
        # though naive floating-point division of (0.3-0.0)/0.1 comes out
        # just under 3.0. A weights vector must match that true count.
        c_dec = Continuous(0.0, 0.3, 0.1)
        @test length(c_dec.values) == 4
        @test_throws ArgumentError Continuous(0.0, 0.3, 0.1; weights=[1.0, 1.0, 1.0])
        @test Continuous(0.0, 0.3, 0.1; weights=[1.0, 1.0, 1.0, 1.0]) isa Hyperopt.Domain
    end

    @testset "values-based construction (Domain replaces a plain candidate array)" begin
        rng = StableRNG(1)

        nom_v = Nominal([true, false]) # unordered -- Nominal never checks order
        @test length(nom_v.values) == 2
        @test all(rand(rng, nom_v) isa Bool for _ in 1:100)
        @test all(rand(rng, nom_v) in nom_v for _ in 1:100)

        nom_fns = Nominal([tanh, exp, identity]) # functions have no natural order -- Nominal, not Ordinal
        @test length(nom_fns.values) == 3
        @test all(rand(rng, nom_fns) in (tanh, exp, identity) for _ in 1:100)
        @test all(rand(rng, nom_fns) in nom_fns for _ in 1:100)

        # log-spaced (non-uniform, but genuinely increasing) numeric values --
        # Continuous, not Ordinal, since this is a numeric parameter with a
        # relatively high point count.
        logspaced = Continuous(exp10.(LinRange(-1, 3, 50)))
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

        # Regression: membership is exact equality against d.values, not an
        # isapprox with some fixed tolerance -- a fixed tolerance would be
        # wrong for small-magnitude domains (e.g. tiny learning rates), where
        # the true grid spacing can be far smaller than any fixed absolute
        # tolerance floor, making an off-grid midpoint falsely test as a member.
        tiny = Continuous(exp10.(range(-10, -5, length=50)))
        midpoint = (tiny.values[1] + tiny.values[2]) / 2
        @test midpoint ∉ tiny

        # `missing in d` is always a definite Bool (false), never Julia's
        # usual 3-valued `missing`-propagating comparison result -- for every
        # domain kind, including a level-count-only one (whose Base.OneTo
        # membership check would otherwise fall into Julia's generic
        # missing-propagating `in` fallback).
        @test (missing in nom) === false
        @test (missing in nom_v) === false
        @test (missing in c) === false
    end
end
