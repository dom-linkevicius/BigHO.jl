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

        @test Nominal([tanh, exp]).values == [tanh, exp]
        @test Ordinal([1, 2, 5, 10]).values == [1, 2, 5, 10]

        @test_throws ArgumentError Nominal(0)
        @test_throws ArgumentError Ordinal(0)
        @test_throws ArgumentError Nominal(Int[])
        @test_throws ArgumentError Ordinal(Int[])

        @test_throws ArgumentError Continuous(5, 1)
        @test_throws ArgumentError Continuous(1, 1)
    end

    @testset "Ordinal order checking" begin
        @test_throws ArgumentError Ordinal([1, 5, 2, 10])
        @test_throws ArgumentError Ordinal([3, 2, 1])

        local o_strings
        @test_logs (:warn, r"assuming this is the intended order") begin
            o_strings = Ordinal(["low", "medium", "high"])
        end
        @test o_strings.values == ["low", "medium", "high"]

        @test_logs min_level = Logging.Warn Ordinal(5)
    end

    @testset "transform" begin
        log_d = Continuous(-4, -1; transform=exp10)
        @test BigHO.from_unit(log_d, 0.0) ≈ 1e-4
        @test BigHO.from_unit(log_d, 1.0) ≈ 1e-1
        @test BigHO.from_unit(log_d, 0.5) ≈ exp10(-2.5)

        @test Continuous(0, 2π; transform=sin) isa BigHO.Domain

        @test_throws ArgumentError Continuous(0, 1; transform=x -> 1 / x)
        @test_throws ArgumentError Continuous(0, Inf)
    end

    @testset "from_unit" begin
        levels = Nominal([:a, :b, :c, :d])

        @test BigHO.from_unit(levels, 0.0) === :a
        @test BigHO.from_unit(levels, 1.0) === :d
        @test [BigHO.from_unit(levels, (i - 0.5) / 4) for i in 1:4] == [:a, :b, :c, :d]

        @test BigHO.from_unit(levels, 0.25) === :a
        @test BigHO.from_unit(levels, nextfloat(0.25)) === :b

        @test BigHO.from_unit(c, 0.0) == 1.0
        @test BigHO.from_unit(c, 1.0) == 5.0
        @test BigHO.from_unit(c, 0.25) == 2.0

        @test BigHO.from_unit(levels, -1.0) === :a
        @test BigHO.from_unit(levels, 2.0) === :d
        @test BigHO.from_unit(c, -1.0) == 1.0
        @test BigHO.from_unit(c, 3.0) == 5.0
    end

    @testset "length" begin
        @test length(nom) == 4
        @test length(ord) == 3
        @test length(Nominal([tanh, exp])) == 2
        @test_throws MethodError length(c)
    end
end
