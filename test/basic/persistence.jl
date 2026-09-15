@testset "Persistence" begin
    @info "Testing Persistence"

    ho_novalidate = Hyperoptimizer(p -> p.a, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws ArgumentError run!(ho_novalidate; save_every=1)
    @test_throws ArgumentError run!(ho_novalidate; save_every=0, save_path="unused.jld2")

    ho_noobjective = Hyperoptimizer(nothing, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws ArgumentError run!(ho_noobjective; save_path="unused.jld2")

    ho_nosave = Hyperoptimizer(p -> p.a^2, (a=Ordinal(1:5),); n=5)
    run!(ho_nosave)
    @test length(results(ho_nosave)) == 5

    mktempdir() do dir
        path = joinpath(dir, "oneoff.jld2")
        ho_oneoff = Hyperoptimizer(p -> p.a^2, (a=Ordinal(1:10),); n=4)
        run!(ho_oneoff)
        save_hyperoptimizer(ho_oneoff, path)
        @test isfile(path)
        @test !isfile(path * ".tmp")

        reloaded = load_hyperoptimizer(p -> p.a^2, path)
        @test length(reloaded.runs) == length(ho_oneoff.runs)
        @test results(reloaded) == results(ho_oneoff)
        @test minimum(reloaded) == minimum(ho_oneoff)
        @test reloaded.status == ho_oneoff.status

        ho_partial = Hyperoptimizer(p -> p.a^2, (a=Ordinal(1:10),); n=6)
        BigHO.ask!(ho_partial)
        save_hyperoptimizer(ho_partial, path)
        @test load_hyperoptimizer(p -> p.a^2, path).n_pending == 1
    end

    mktempdir() do dir
        path = joinpath(dir, "checkpoint.jld2")

        ho = Hyperoptimizer(p -> p.a^2, (a=Ordinal(1:20),); n=10)
        run!(ho; executor=Serial(), save_every=3, save_path=path)
        @test length(results(ho)) == 10
        @test ho.status == BigHO.Finished
        @test isfile(path)
        @test !isfile(path * ".tmp")
        @test length(readdir(dir)) == 1

        loaded = load_hyperoptimizer(p -> p.a^2, path)
        @test loaded.status == BigHO.Finished
        @test loaded.n_pending == 0
        @test loaded.n == ho.n
        @test length(loaded.runs) == length(ho.runs) == 10
        @test [e.value for e in loaded.runs] == [e.value for e in ho.runs]
        @test minimum(loaded) == minimum(ho)

        @test_logs (:warn, r"already reached its target") run!(loaded)
    end

    mktempdir() do dir
        path = joinpath(dir, "checkpoint.jld2")

        ho = Hyperoptimizer(p -> p.a^2, (a=Ordinal(1:20),); n=6)
        run!(ho; executor=Serial(), save_path=path)
        @test isfile(path)
        @test !isfile(path * ".tmp")
        loaded = load_hyperoptimizer(p -> p.a^2, path)
        @test loaded.status == BigHO.Finished
        @test length(loaded.runs) == 6
        @test [e.value for e in loaded.runs] == [e.value for e in ho.runs]
    end

    mktempdir() do dir
        path = joinpath(dir, "checkpoint.jld2")

        g(p) = (p.a - 7)^2
        ho = Hyperoptimizer(g, (a=Ordinal(0:10),); n=5)
        run!(ho; executor=Serial(), save_path=path)
        @test length(ho.runs) == 5

        loaded = load_hyperoptimizer(g, path)
        @test loaded.n == 5
        settarget!(loaded, 200)
        run!(loaded)
        @test loaded.status == BigHO.Finished
        @test length(loaded.runs) == 200
        @test length(results(loaded)) == 200
        @test minimum(loaded) == 0
        @test minimizer(loaded) == [7]
    end
end
