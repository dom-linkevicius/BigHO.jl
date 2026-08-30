@testset "Persistence" begin
    @info "Testing Persistence"

    # save_every requires save_path (nowhere to save to otherwise); save_path
    # alone is valid -- it just means "save once, at the end" (see below).
    ho_novalidate = Hyperoptimizer(a -> a, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws ArgumentError run!(ho_novalidate; save_every=1)
    @test_throws ArgumentError run!(ho_novalidate; save_every=0, save_path="unused.jld2")

    # save_path substitutes `nothing` for the objective in the saved file --
    # ambiguous (and useless to resume) if the real objective already was
    # `nothing`, so this is rejected up front rather than silently
    # checkpointing something load_hyperoptimizer could never usefully undo.
    ho_noobjective = Hyperoptimizer(nothing, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws ArgumentError run!(ho_noobjective; save_path="unused.jld2")

    # Omitting both leaves behavior exactly as before -- no file, no error.
    ho_nosave = Hyperoptimizer(a -> a^2, (a=Ordinal(1:5),); n=5)
    run!(ho_nosave)
    @test length(results(ho_nosave)) == 5

    mktempdir() do dir
        path = joinpath(dir, "checkpoint.jld2")

        # save_every=3 over n=10 (Serial: one trial resolved per outer-loop
        # iteration, so each multiple of 3 is hit exactly) checkpoints after
        # the 3rd, 6th, and 9th trial told -- but a final checkpoint is
        # always taken once the run ends regardless, so the file on disk
        # ends up reflecting the TRUE final state (all 10, Finished), not
        # stuck at the last save_every boundary. The objective itself is an
        # anonymous closure throughout this file -- proving it never needs
        # to survive serialization (see load_hyperoptimizer below). The file
        # itself is always the single overwritten path -- no numbered
        # variants -- and the atomic-rename temp file never lingers after a
        # successful save.
        ho = Hyperoptimizer(a -> a^2, (a=Ordinal(1:20),); n=10)
        run!(ho; executor=Serial(), save_every=3, save_path=path)
        @test length(results(ho)) == 10
        @test ho.status == BigHO.Finished
        @test isfile(path)
        @test !isfile(path * ".tmp")
        @test length(readdir(dir)) == 1 # single-file-overwrite: nothing else was ever created here

        # Reloading needs the objective supplied fresh -- it was never part
        # of the checkpoint at all (not merely dropped-then-restorable), so
        # a plain equivalent closure is enough; nothing ties it back to the
        # original object.
        loaded = load_hyperoptimizer(a -> a^2, path)
        @test loaded.status == BigHO.Finished # the final, always-taken checkpoint, not a mid-run one
        @test loaded.n_pending == 0
        @test loaded.n == ho.n
        @test length(loaded.runs) == length(ho.runs) == 10
        @test [e.value for e in loaded.runs] == [e.value for e in ho.runs]
        @test minimum(loaded) == minimum(ho)

        # Calling run! again on the fully-finished, reloaded copy is the
        # already-tested "nothing to do" no-op-with-warning path -- confirms
        # the reloaded status round-trips meaningfully, not just cosmetically.
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

        # The actual point of save_path/load_hyperoptimizer together: a
        # finished, saved run can be loaded back in what's effectively a
        # fresh session (a fresh objective closure, a fresh Hyperoptimizer
        # object) and genuinely resumed past its original target via
        # settarget! -- not just round-tripped, but continued to keep
        # fitting. n starts far too small to find the true optimum, then
        # settarget! raises it enough that the resumed run reliably does.
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
