@testset "Domains" begin
    @info "Testing Domain primitives"

    @test Levels <: Ordinal
    @test Continuous <: Ordinal
    @test !(Categorical <: Ordinal) # nominal/unordered -- deliberately not part of the Ordinal hierarchy

    lv = Levels(4)
    @test lv isa Domain
    @test lv isa Ordinal
    @test lv.levels == 4

    cat = Categorical(3)
    @test cat isa Domain
    @test !(cat isa Ordinal)
    @test cat.levels == 3

    c = Continuous(1, 5, 1) # dt=1 -> grid points 1,2,3,4,5
    @test c isa Ordinal
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

    @testset "rand, uniform (no weights)" begin
        rng = StableRNG(1)

        @test all(rand(rng, lv) in 1:4 for _ in 1:1000)
        @test all(rand(rng, cat) in 1:3 for _ in 1:1000)
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

        @test rand(lv) isa Int   # no explicit rng also works
        @test rand(cat) isa Int
        @test rand(c) isa Float64
    end

    @testset "rand, weighted" begin
        rng = StableRNG(1)

        # Weight everything onto a single level/point -- draws must always land there.
        lv_w = Levels(4; weights=[0.0, 1.0, 0.0, 0.0])
        @test all(rand(rng, lv_w) == 2 for _ in 1:200)

        cat_w = Categorical(3; weights=[1.0, 0.0, 0.0])
        @test all(rand(rng, cat_w) == 1 for _ in 1:200)

        c_w = Continuous(1, 5, 1; weights=[0.0, 0.0, 0.0, 0.0, 1.0]) # forces the last grid point, 5.0
        @test all(rand(rng, c_w) == 5.0 for _ in 1:200)

        # Wrong-length weights are rejected at construction, not silently at sample time.
        @test_throws ArgumentError Levels(4; weights=[1.0, 2.0])
        @test_throws ArgumentError Categorical(3; weights=[1.0, 2.0, 3.0, 4.0])
        @test_throws ArgumentError Continuous(1, 5, 1; weights=[1.0, 2.0])
    end

    @testset "values-based construction (Domain replaces a plain candidate array)" begin
        rng = StableRNG(1)

        lv_v = Levels([true, false])
        @test lv_v.levels == 2
        @test all(rand(rng, lv_v) isa Bool for _ in 1:100)
        @test all(rand(rng, lv_v) in lv_v for _ in 1:100)

        cat_v = Categorical([tanh, exp, identity])
        @test cat_v.levels == 3
        @test all(rand(rng, cat_v) in (tanh, exp, identity) for _ in 1:100)
        @test all(rand(rng, cat_v) in cat_v for _ in 1:100)

        # log-spaced (non-uniform) values: only representable via Levels, not Continuous.
        logspaced = Levels(exp10.(LinRange(-1, 3, 50)))
        @test all(rand(rng, logspaced) in logspaced for _ in 1:200)

        # Domains constructed from a plain level count (no values) don't carry actual
        # candidate values -- membership only checks against the index range 1:levels.
        @test 1 in lv
        @test 5 ∉ lv
        @test "low" ∉ lv # non-integer -- never a member of an index-only domain

        # Continuous membership: on-grid values are members, off-grid values are not.
        @test 1.0 in c
        @test 3.0 in c
        @test 3.5 ∉ c   # c has dt=1, so 3.5 isn't on the grid
        @test 6.0 ∉ c   # out of [min,max]
    end
end
