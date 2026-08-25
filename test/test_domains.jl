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
end
