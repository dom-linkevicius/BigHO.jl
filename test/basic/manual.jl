@testset "Manual" begin
    @info "Testing Manual"

    # No macro, no `run!` -- just construct and draw samples directly via `ask!`.
    # `objective` is `nothing` here since this testset only exercises sampling,
    # never evaluation.
    ho = Hyperoptimizer(nothing, (a=Continuous(1, 2, 1 / 49), b=Nominal([true, false]), c=Nominal(randn(100))); n=10)
    show(devnull, ho)
    for _ in 1:10
        entry = BigHO.ask!(ho)
        println(entry.id, "\t", entry.params)
    end
    @test length(ho.runs) == 10
    show(devnull, ho)

    ho2 = Hyperoptimizer(nothing, (a=Continuous(1, 2, 1 / 49), b=Nominal([true, false]), c=Nominal(randn(100))); n=10)
    entries = [BigHO.ask!(ho2) for _ in 1:10]
    @test length(entries) == 10
    @test length(ho2.runs) == 10
    @test all(e -> e.params[1] in ho2.candidates[1], entries)

    # With zero completed runs (only `ask!`, never `tell!`), the optimum accessors
    # must not silently return a sentinel -- they throw instead.
    @test_throws ErrorException minimum(ho2)
    @test_throws ErrorException minimizer(ho2)

    # ask! enforces ho.n itself -- not just run!'s driver loop -- so manual
    # ask!/tell! callers can't silently draw past the declared target.
    ho3 = Hyperoptimizer(nothing, (a=Nominal([1, 2, 3]),); n=2)
    BigHO.ask!(ho3)
    BigHO.ask!(ho3)
    @test_throws ArgumentError BigHO.ask!(ho3)
end
