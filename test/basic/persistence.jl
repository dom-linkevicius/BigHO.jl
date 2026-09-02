@testset "Persistence" begin
    @info "Testing Persistence"

    # save_every requires save_path (nowhere to save to otherwise); save_path
    # alone is valid -- it just means "save once, at the end" (see below).
    ho_novalidate = Hyperoptimizer(a -> a, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws ArgumentError run!(ho_novalidate; save_every=1)
    @test_throws ArgumentError run!(ho_novalidate; save_every=0, save_path="unused.jld2")

    # save_path substitutes `nothing` for the objective -- rejected up front if the real one already was `nothing`, since that'd be ambiguous to resume.
    ho_noobjective = Hyperoptimizer(nothing, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws ArgumentError run!(ho_noobjective; save_path="unused.jld2")

    # Omitting both leaves behavior exactly as before -- no file, no error.
    ho_nosave = Hyperoptimizer(a -> a^2, (a=Ordinal(1:5),); n=5)
    run!(ho_nosave)
    @test length(results(ho_nosave)) == 5

    mktempdir() do dir
        path = joinpath(dir, "checkpoint.jld2")

        # save_every=3 checkpoints periodically, but a final checkpoint always reflects the TRUE end state (all 10, Finished).
        # Single overwritten file, no numbered variants; the atomic-rename temp file never lingers after a successful save.
        ho = Hyperoptimizer(a -> a^2, (a=Ordinal(1:20),); n=10)
        run!(ho; executor=Serial(), save_every=3, save_path=path)
        @test length(results(ho)) == 10
        @test ho.status == BigHO.Finished
        @test isfile(path)
        @test !isfile(path * ".tmp")
        @test length(readdir(dir)) == 1 # single-file-overwrite: nothing else was ever created here

        # Reloading needs the objective supplied fresh -- it was never part of the checkpoint at all.
        loaded = load_hyperoptimizer(a -> a^2, path)
        @test loaded.status == BigHO.Finished # the final, always-taken checkpoint, not a mid-run one
        @test loaded.n_pending == 0
        @test loaded.n == ho.n
        @test length(loaded.runs) == length(ho.runs) == 10
        @test [e.value for e in loaded.runs] == [e.value for e in ho.runs]
        @test minimum(loaded) == minimum(ho)

        # Confirms the reloaded status round-trips meaningfully -- run! treats it as already-finished.
        @test_logs (:warn, r"already reached its target") run!(loaded)
    end

    mktempdir() do dir
        path = joinpath(dir, "checkpoint.jld2")

        # save_path alone (no save_every): saves exactly once, at the end.
        ho = Hyperoptimizer(a -> a^2, (a=Ordinal(1:20),); n=6)
        run!(ho; executor=Serial(), save_path=path)
        @test isfile(path)
        @test !isfile(path * ".tmp")
        loaded = load_hyperoptimizer(a -> a^2, path)
        @test loaded.status == BigHO.Finished
        @test length(loaded.runs) == 6
        @test [e.value for e in loaded.runs] == [e.value for e in ho.runs]
    end

    mktempdir() do dir
        path = joinpath(dir, "checkpoint.jld2")

        # The actual point of save_path/load_hyperoptimizer: a finished run can be reloaded and genuinely resumed past its original target via settarget!.
        # n starts too small to find the true optimum; settarget! raises it enough that the resumed run reliably does.
        g(a) = (a - 7)^2
        ho = Hyperoptimizer(g, (a=Ordinal(0:10),); n=5)
        run!(ho; executor=Serial(), save_path=path)
        @test length(ho.runs) == 5
        @test minimum(ho) > 0 # too few trials yet to have hit the true optimum

        loaded = load_hyperoptimizer(g, path)
        @test loaded.n == 5
        settarget!(loaded, 200)
        run!(loaded)
        @test loaded.status == BigHO.Finished
        @test length(loaded.runs) == 200
        @test length(results(loaded)) == 200
        @test minimum(loaded) == 0 # 200 trials over 11 candidates finds it with overwhelming probability
        @test minimizer(loaded) == [7]
    end
end
