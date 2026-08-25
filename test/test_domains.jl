@testset "Domains" begin
    @info "Testing Domain primitives"

    c = Continuous(1, 5)
    @test c isa Domain
    @test c.min == 1.0
    @test c.max == 5.0
    @test c.log == false

    clog = Continuous(1e-4, 1e2; log=true)
    @test clog.log == true
    @test clog.min == 1e-4
    @test clog.max == 1e2

    o = Ordinal(4)
    @test o isa Domain
    @test o.levels == 4

    cat = Categorical(3)
    @test cat isa Domain
    @test cat.levels == 3

    @test !(Ordinal(3) isa Categorical)
    @test !(Categorical(3) isa Ordinal)

    @testset "rand" begin
        rng = StableRNG(1)

        # Continuous: draws fall within [min, max], both with and without `log`.
        @test all(1:1000) do _
            v = rand(rng, c)
            1.0 <= v <= 5.0
        end
        @test all(1:1000) do _
            v = rand(rng, clog)
            1e-4 <= v <= 1e2
        end
        @test rand(c) isa Float64 # no explicit rng also works

        # log=true requires a strictly positive lower bound.
        @test_throws ArgumentError rand(rng, Continuous(-1, 5; log=true))
        @test_throws ArgumentError rand(rng, Continuous(0, 5; log=true))

        # Ordinal/Categorical: draws are level indices in 1:levels.
        @test all(rand(rng, o) in 1:4 for _ in 1:1000)
        @test all(rand(rng, cat) in 1:3 for _ in 1:1000)
        @test rand(o) isa Int
        @test rand(cat) isa Int
    end
end
