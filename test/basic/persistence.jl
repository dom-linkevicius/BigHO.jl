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
        reloaded_partial = load_hyperoptimizer(p -> p.a^2, path)
        @test reloaded_partial.n_pending == 0
        @test reloaded_partial.runs[1].status === BigHO.Abandoned
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

@testset "SuccessiveHalving bracket state survives a checkpoint" begin
    @info "Testing that a checkpointed Hyperband/ASHA restores its active brackets and can be resumed"

    toy(p; pre_artefact=nothing) = ((p.a - 3.0)^2 + 1.0 / p.r, nothing)

    for sampler in (Hyperband(R=9, η=3, r_min=1, iterations=2), ASHA(R=9, η=3, r_min=1, iterations=2))
        mktempdir() do dir
            path = joinpath(dir, "sh.jld2")
            ho = Hyperoptimizer(Stateful(toy), (a=Continuous(0, 10),), sampler)
            run!(ho; executor=Serial(), save_every=1, save_path=path)

            loaded = load_hyperoptimizer(Stateful(toy), path)
            @test loaded.status == BigHO.Finished
            @test length(loaded.runs) == ho.n
            @test loaded.sampler.current_itr[] == ho.sampler.current_itr[]
            @test loaded.sampler.last_bracket[] == ho.sampler.last_bracket[]
            @test length(loaded.sampler.active_brackets) == length(ho.sampler.active_brackets)
            for (restored, original) in zip(loaded.sampler.active_brackets, ho.sampler.active_brackets)
                @test restored.iteration == original.iteration
                @test restored.bracket == original.bracket
                @test [r.ids for r in restored.rungs] == [r.ids for r in original.rungs]
                @test [r.capacity for r in restored.rungs] == [r.capacity for r in original.rungs]
                @test [r.resource for r in restored.rungs] == [r.resource for r in original.rungs]
            end
        end
    end
end

@testset "SuccessiveHalving resumes mid-run from a checkpoint" begin
    @info "Testing that a Hyperband checkpoint taken mid-run continues the same schedule"

    toy(p; pre_artefact=nothing) = ((p.a - 3.0)^2 + 1.0 / p.r, nothing)

    mktempdir() do dir
        path = joinpath(dir, "partial.jld2")
        sampler = Hyperband(R=9, η=3, r_min=1)
        ho = Hyperoptimizer(Stateful(toy), (a=Continuous(0, 10),), sampler)
        for _ in 1:5
            entry = BigHO.ask!(ho)
            BigHO.tell!(ho, entry, BigHO.call_objective(ho.objective, entry.params, entry.pre_artefact))
        end
        save_hyperoptimizer(ho, path)
        @test length(ho.sampler.active_brackets) == 1

        loaded = load_hyperoptimizer(Stateful(toy), path)
        @test length(loaded.runs) == 5
        @test loaded.sampler.active_brackets[1].rungs[1].ids == collect(1:5)

        run!(loaded; executor=Serial())
        @test loaded.status == BigHO.Finished
        @test length(loaded.runs) == loaded.n
        @test sort([e.id for e in loaded.runs]) == collect(1:loaded.n)
    end
end

@testset "Trials still in flight at checkpoint time are abandoned on load" begin
    @info "Testing that a checkpoint written mid-flight doesn't leave a resumed run waiting forever"

    toy(p; pre_artefact=nothing) = ((p.a - 3.0)^2 + 1.0 / p.r, nothing)

    mktempdir() do dir
        path = joinpath(dir, "inflight.jld2")
        ho = Hyperoptimizer(Stateful(toy), (a=Continuous(0, 10),), Hyperband(R=9, η=3, r_min=1))
        for _ in 1:9
            BigHO.ask!(ho)
        end
        for i in 1:7
            BigHO.tell!(ho, ho.runs[i], BigHO.call_objective(ho.objective, ho.runs[i].params, ho.runs[i].pre_artefact))
        end
        save_hyperoptimizer(ho, path)
        @test ho.n_pending == 2
        @test BigHO.blocked(ho.sampler, ho)

        loaded = @test_logs (:warn, r"still in flight") match_mode = :any load_hyperoptimizer(Stateful(toy), path)
        @test loaded.n_pending == 0
        @test count(e -> e.status === BigHO.Abandoned, loaded.runs) == 2
        @test !BigHO.blocked(loaded.sampler, loaded)

        run!(loaded; executor=Serial())
        @test loaded.status == BigHO.Finished
        @test length(loaded.runs) == loaded.n
        @test count(e -> e.status === BigHO.Abandoned, loaded.runs) == 2
    end
end
