@testset "Domains" begin
    @info "Testing Domain primitives"

    c = Continuous(1, 5)
    @test c isa Domain
    @test c.min == 1.0
    @test c.max == 5.0

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

        # Continuous: draws fall within [min, max].
        @test all(1:1000) do _
            v = rand(rng, c)
            1.0 <= v <= 5.0
        end
        @test rand(c) isa Float64 # no explicit rng also works

        # Ordinal/Categorical: draws are level indices in 1:levels.
        @test all(rand(rng, o) in 1:4 for _ in 1:1000)
        @test all(rand(rng, cat) in 1:3 for _ in 1:1000)
        @test rand(o) isa Int
        @test rand(cat) isa Int
    end
end
