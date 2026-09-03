@testset "Hyperband/ASHA construction and basic run" begin
    @info "Testing Hyperband/ASHA construction and basic run"

    toy(r, a, b) = (a - 3.0)^2 + (b - 1.0)^2 + 1.0 / r

    for (name, sampler) in (("ASHA", ASHA(R=81, η=3, r_min=1)), ("Hyperband", Hyperband(R=81, η=3, r_min=1)))
        ho = Hyperoptimizer(toy, (a=Continuous(0, 10, 0.1), b=Continuous(0, 5, 0.1)), sampler; n=150)
        run!(ho; show_progress=false)
        @test length(ho.runs) == 150
        @test length(ho.completed) == 150 # toy() never fails/NaNs
        @test ho.status == BigHO.Finished

        # :r is a real candidate, prepended -- so it shows up in params/minimizer/history like any other.
        @test ho.params[1] == :r
        r_values = [e.params.r for e in ho.runs]
        @test all(r -> r in ho.candidates[1], r_values)

        # Space-filling isn't the point here, but the resource schedule should visibly
        # favor the bottom rung -- most trials are cheap early exits, few reach R.
        counts = Dict(r => count(==(r), r_values) for r in unique(r_values))
        @test counts[minimum(r_values)] > counts[maximum(r_values)]

        # An easy, low-dimensional problem -- both samplers should get reasonably close.
        m = minimizer(ho)
        @test abs(m[2] - 3.0) < 1.0
        @test abs(m[3] - 1.0) < 1.0
    end
end

@testset "Hyperband/ASHA warm-start correctness" begin
    @info "Testing Hyperband/ASHA warm-start (pre_artefact/post_artefact) correctness"

    # post_artefact = (this call's own sequence number, the sequence number it resumed
    # from or nothing) -- objectives never learn their own trial id, so a monotonically
    # increasing call counter is the tag used to identify "which call produced this
    # artefact" below. A promoted trial's pre_artefact must exactly match a real, earlier,
    # Completed trial's post_artefact -- never dangling, never pointing forward in time.
    for sampler in (ASHA(R=27, η=3, r_min=1), Hyperband(R=27, η=3, r_min=1))
        call_id = Ref(0)
        function stateful_obj(r, a, b; pre_artefact=nothing)
            call_id[] += 1
            resumed_from = pre_artefact === nothing ? nothing : pre_artefact[1]
            loss = a + 0.001 * b
            return loss, (call_id[], resumed_from)
        end

        ho = Hyperoptimizer(Stateful(stateful_obj), (a=Nominal([0.0, 1.0, 2.0]), b=Nominal([0.0, 1.0])), sampler; n=60)
        run!(ho; show_progress=false)

        n_promoted = 0
        for e in ho.runs
            e.status == BigHO.Completed || continue
            if e.pre_artefact !== nothing
                n_promoted += 1
                pre_call_id, _ = e.pre_artefact
                @test pre_call_id !== nothing
                producer = findfirst(x -> x.status == BigHO.Completed && x.post_artefact[1] == pre_call_id, ho.runs)
                @test producer !== nothing
                @test producer < e.id # causality: can only resume something already told
            end
        end
        @test n_promoted > 0 # otherwise this test isn't exercising promotion at all
    end
end

@testset "Hyperband/ASHA reserved :r and failure handling" begin
    @info "Testing Hyperband/ASHA reserved :r name and NaN/failure exclusion"

    @test_throws ArgumentError Hyperoptimizer(a -> a, (r=Nominal([1]), a=Nominal([1])), ASHA(R=9, η=3, r_min=1); n=5)
    @test_throws ArgumentError Hyperoptimizer(a -> a, (r=Nominal([1]), a=Nominal([1])), Hyperband(R=9, η=3, r_min=1); n=5)

    # A trial that fails must never corrupt rung bookkeeping for the ones that succeed.
    let n_calls = Ref(0)
        global hb_flaky(r, a) = (n_calls[] += 1; n_calls[] % 5 == 0 ? NaN : a + 1.0 / r)
    end
    ho = Hyperoptimizer(hb_flaky, (a=Nominal([1, 2, 3, 4, 5]),), ASHA(R=9, η=3, r_min=1); n=50)
    @test_logs (:warn, r"NaN") match_mode = :any run!(ho; show_progress=false)
    n_failed = count(e -> e.status == BigHO.Failed, ho.runs)
    @test n_failed > 0
    @test length(ho.completed) == length(ho.runs) - n_failed
end

@testset "Hyperband/ASHA settarget! resume" begin
    @info "Testing Hyperband/ASHA settarget!/resume (neither is a FixedPlanSampler)"

    for sampler in (ASHA(R=9, η=3, r_min=1), Hyperband(R=9, η=3, r_min=1))
        ho = Hyperoptimizer((r, a) -> a, (a=Nominal([1, 2, 3]),), sampler; n=10)
        run!(ho; show_progress=false)
        @test length(ho.runs) == 10
        settarget!(ho, 20)
        run!(ho; show_progress=false)
        @test length(ho.runs) == 20
    end
end

@testset "Hyperband/ASHA under Threaded executor" begin
    @info "Testing Hyperband/ASHA under the Threaded executor (sync rung-barrier under real concurrency)"

    toy(r, a, b) = (a - 3.0)^2 + (b - 1.0)^2 + 1.0 / r

    for sampler in (ASHA(R=81, η=3, r_min=1), Hyperband(R=81, η=3, r_min=1))
        ho = Hyperoptimizer(toy, (a=Continuous(0, 10, 0.1), b=Continuous(0, 5, 0.1)), sampler; n=200)
        run!(ho; executor=Threaded(8), show_progress=false)
        @test length(ho.runs) == 200
        @test length(ho.completed) == 200
        @test ho.n_pending == 0
        @test ho.status == BigHO.Finished
        @test sort([e.id for e in ho.runs]) == collect(1:200) # no id gaps/dupes under real concurrency
    end
end
