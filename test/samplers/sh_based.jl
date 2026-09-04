@testset "Hyperband/ASHA constructor validation" begin
    @info "Testing SuccessiveHalving's R/η/r_min validation"

    @test_throws ArgumentError Hyperband(R=0, η=3, r_min=1)
    @test_throws ArgumentError Hyperband(R=-1, η=3, r_min=1)
    @test_throws ArgumentError Hyperband(R=9, η=1, r_min=1)
    @test_throws ArgumentError Hyperband(R=9, η=0, r_min=1)
    @test_throws ArgumentError Hyperband(R=9, η=-3, r_min=1)
    @test_throws ArgumentError Hyperband(R=9, η=3, r_min=0)
    @test_throws ArgumentError Hyperband(R=9, η=3, r_min=-1)
    @test_throws ArgumentError Hyperband(R=1, η=3, r_min=9) # r_min > R

    Hyperband(R=9, η=2, r_min=1) # η=2 is the smallest valid value
    Hyperband(R=9, η=3, r_min=9) # r_min == R is valid (degenerate single-bracket, single-rung)
end

@testset "Hyperband/ASHA construction and basic run" begin
    @info "Testing Hyperband/ASHA construction and basic run"

    toy(r, a, b) = (a - 3.0)^2 + (b - 1.0)^2 + 1.0 / r

    for (name, sampler) in (("Hyperband", Hyperband(R=81, η=3, r_min=1)), ("ASHA", ASHA(R=81, η=3, r_min=1)))
        ho = Hyperoptimizer(toy, (a=Continuous(0, 10, 0.1), b=Continuous(0, 5, 0.1)), sampler)
        run!(ho; show_progress=false)
        @test ho.n == length(ho.runs) # the one-pass total is self-determined, not passed in
        @test length(ho.completed) == ho.n # toy() never fails/NaNs
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
    for sampler in (Hyperband(R=27, η=3, r_min=1), ASHA(R=27, η=3, r_min=1))
        call_id = Ref(0)
        function stateful_obj(r, a, b; pre_artefact=nothing)
            call_id[] += 1
            resumed_from = pre_artefact === nothing ? nothing : pre_artefact[1]
            loss = a + 0.001 * b
            return loss, (call_id[], resumed_from)
        end

        ho = Hyperoptimizer(Stateful(stateful_obj), (a=Nominal([0.0, 1.0, 2.0]), b=Nominal([0.0, 1.0])), sampler)
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

@testset "Hyperband/ASHA reserved :r, explicit n, and failure handling" begin
    @info "Testing Hyperband/ASHA reserved :r name, explicit-n rejection, and NaN/failure exclusion"

    @test_throws ArgumentError Hyperoptimizer(a -> a, (r=Nominal([1]), a=Nominal([1])), Hyperband(R=9, η=3, r_min=1))
    @test_throws ArgumentError Hyperoptimizer(a -> a, (r=Nominal([1]), a=Nominal([1])), ASHA(R=9, η=3, r_min=1))

    # n is fully determined by R/η/r_min -- passing it explicitly is rejected, not silently ignored.
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Nominal([1]),), Hyperband(R=9, η=3, r_min=1); n=5)
    @test_throws ArgumentError Hyperoptimizer(a -> a, (a=Nominal([1]),), ASHA(R=9, η=3, r_min=1); n=5)

    # A trial that fails must never corrupt rung bookkeeping for the ones that succeed.
    n_calls = Ref(0)
    global hb_flaky(r, a) = (n_calls[] += 1; n_calls[] % 5 == 0 ? NaN : a + 1.0 / r)
    for sampler in (Hyperband(R=9, η=3, r_min=1), ASHA(R=9, η=3, r_min=1))
        n_calls[] = 0
        ho = Hyperoptimizer(hb_flaky, (a=Nominal([1, 2, 3, 4, 5]),), sampler)
        @test_logs (:warn, r"NaN") match_mode = :any run!(ho; show_progress=false)
        n_failed = count(e -> e.status == BigHO.Failed, ho.runs)
        @test n_failed > 0
        @test length(ho.completed) == length(ho.runs) - n_failed
    end
end

@testset "Hyperband/ASHA are FixedPlanSamplers" begin
    @info "Testing Hyperband/ASHA are FixedPlanSamplers (fixed one-pass plan, no settarget!/resume)"

    for sampler in (Hyperband(R=9, η=3, r_min=1), ASHA(R=9, η=3, r_min=1))
        @test sampler isa BigHO.FixedPlanSampler
        ho = Hyperoptimizer((r, a) -> a, (a=Nominal([1, 2, 3]),), sampler)
        run!(ho; show_progress=false)
        @test ho.status == BigHO.Finished
        @test_throws ArgumentError settarget!(ho, ho.n + 10)
    end
end

@testset "Hyperband/ASHA under Threaded executor" begin
    @info "Testing Hyperband/ASHA under the Threaded executor (cross-bracket promotion correctness under real concurrency)"

    toy(r, a, b) = (a - 3.0)^2 + (b - 1.0)^2 + 1.0 / r

    for sampler in (Hyperband(R=81, η=3, r_min=1), ASHA(R=81, η=3, r_min=1))
        ho = Hyperoptimizer(toy, (a=Continuous(0, 10, 0.1), b=Continuous(0, 5, 0.1)), sampler)
        run!(ho; executor=Threaded(8), show_progress=false)
        @test length(ho.runs) == ho.n # the full precomputed total is always reached, even across brackets
        @test length(ho.completed) == ho.n
        @test ho.n_pending == 0
        @test ho.status == BigHO.Finished
        @test sort([e.id for e in ho.runs]) == collect(1:ho.n) # no id gaps/dupes under real concurrency
    end
end

@testset "Hyperband/ASHA under DistributedQueue executor" begin
    @info "Testing Hyperband/ASHA under the DistributedQueue executor (cross-bracket promotion correctness under real concurrency)"

    # n is small: DistributedQueue pays a real process-spawn cost per trial.
    @everywhere using BigHO
    @everywhere sh_dq_toy(r, a, b) = (a - 3.0)^2 + (b - 1.0)^2 + 1.0 / r

    sh_dq_spawn_worker() = first(addprocs(1))
    function sh_dq_setup_worker(pid)
        Distributed.remotecall_eval(Main, [pid], :(begin
            using BigHO
            sh_dq_toy(r, a, b) = (a - 3.0)^2 + (b - 1.0)^2 + 1.0 / r
        end))
        return nothing
    end

    try
        for sampler in (Hyperband(R=9, η=3, r_min=1), ASHA(R=9, η=3, r_min=1))
            ho = Hyperoptimizer(sh_dq_toy, (a=Continuous(0, 10, 0.1), b=Continuous(0, 5, 0.1)), sampler)
            run!(ho; executor=DistributedQueue(3; spawn_worker=sh_dq_spawn_worker, setup_worker=sh_dq_setup_worker), show_progress=false)
            @test length(ho.runs) == ho.n # the full precomputed total is always reached, even across brackets
            @test length(ho.completed) == ho.n
            @test ho.n_pending == 0
            @test ho.status == BigHO.Finished
            @test sort([e.id for e in ho.runs]) == collect(1:ho.n) # no id gaps/dupes under real concurrency
        end
    finally
        rmprocs(filter(!=(1), workers()))
    end
end

@testset "Hyperband rung-level failure handling (shrinkage/abandonment)" begin
    @info "Testing Hyperband's promote-fewer-than-planned and abandon-on-total-wipeout logic"

    # R=9, η=3, r_min=1 -> bracket 3 has capacities rung1=9, rung2=3, rung3=1 (see _capacity).
    s = Hyperband(R=9, η=3, r_min=1)

    function rung1_entries(k, n_total, n_failed)
        runs = BigHO.RunEntry[]
        for idx in 1:n_total
            e = BigHO.RunEntry(idx, (r=1, a=idx), Dict{Symbol,Any}(:rung => 1, :bracket_k => k))
            e = idx <= n_failed ? BigHO._with_result(e, BigHO.Failed, missing, nothing; error=NaN) :
                                   BigHO._with_result(e, BigHO.Completed, Float64(idx), nothing)
            push!(runs, e)
        end
        return runs
    end

    # Shortfall: 7/9 rung-1 trials fail -- only 2 completed, so only 2 (not the planned
    # min(η, ...)=3) are ever promoted, with a warning fired exactly once at the resolving tell!.
    runs = rung1_entries(3, 9, 7)
    logs, _ = Test.collect_test_logs() do
        BigHO.on_tell!(s, runs, runs[end])
    end
    @test count(l -> l.level == Logging.Warn && occursin("promoting fewer than planned", l.message), logs) == 1
    @test BigHO._bracket_decision(s, 3, runs) == (:promote, 3, 1, 8) # best of the 2 completed (id=8, value 8.0 < id=9's 9.0)

    # Promote both survivors -- once the shrunk target (2, not 3) is reached, the bracket must
    # move on to rung 2's own state, not keep waiting for a 3rd promotion that will never come.
    push!(runs, BigHO.RunEntry(10, (r=3, a=8), Dict{Symbol,Any}(:rung => 2, :bracket_k => 3)))
    @test BigHO._bracket_decision(s, 3, runs) == (:promote, 3, 1, 9)
    push!(runs, BigHO.RunEntry(11, (r=3, a=9), Dict{Symbol,Any}(:rung => 2, :bracket_k => 3)))
    @test BigHO._bracket_decision(s, 3, runs) == (:wait,) # both Pending now, not stuck asking for a nonexistent 3rd

    # Total wipeout: every rung-1 trial fails -- the bracket is abandoned (warned once) and
    # control moves to bracket 2's own fresh draw, instead of blocking forever.
    wiped = rung1_entries(3, 9, 9)
    logs2, _ = Test.collect_test_logs() do
        BigHO.on_tell!(s, wiped, wiped[end])
    end
    @test count(l -> l.level == Logging.Warn && occursin("abandoning bracket", l.message), logs2) == 1
    @test BigHO._bracket_decision(s, 3, wiped) == (:draw, 2)

    # End-to-end through run!, under both Serial and Threaded: failures depend only on the
    # candidate value (never on dispatch order or timing), so both executors reach the exact
    # same outcome -- and neither hangs despite one bracket falling short of its full plan.
    flaky(r, a) = a > 4 ? NaN : Float64(a) + 1.0 / r
    for (label, executor) in (("Serial", Serial()), ("Threaded", Threaded(4)))
        ho = Hyperoptimizer(flaky, (a=Nominal(collect(1:20)),), Hyperband(R=27, η=3, r_min=1))
        logs3, _ = Test.collect_test_logs() do
            run!(ho; executor=executor, show_progress=false)
        end
        @test ho.status == BigHO.Finished # completes despite falling short of the full plan -- never hangs
        n_failed = count(e -> e.status == BigHO.Failed, ho.runs)
        @test n_failed == 32
        @test length(ho.runs) == 65 # short of ho.n=69 -- one bracket's rungs shrank below their planned capacity
        @test length(ho.completed) == length(ho.runs) - n_failed
        @test count(l -> l.level == Logging.Warn && occursin("promoting fewer than planned", l.message), logs3) == 1
    end
end
