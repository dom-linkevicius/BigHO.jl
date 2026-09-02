# Produces candidates 1, 2, 3, ... in order, so a specific trial can be rigged by VALUE rather than by timing.
mutable struct SequentialSampler <: BigHO.Sampler
    n_calls::Int
end
SequentialSampler() = SequentialSampler(0)
function (s::SequentialSampler)(ctx)
    s.n_calls += 1
    return [s.n_calls]
end
BigHO.init!(::SequentialSampler, ctx) = nothing
BigHO.exhausted(::SequentialSampler, ho) = false
BigHO.on_tell!(::SequentialSampler, entry) = nothing

@testset "DistributedQueue executor" begin
    @info "Testing DistributedQueue executor"

    @test_throws ArgumentError DistributedQueue(0; spawn_worker=() -> error("should not be called"))
    @test_throws ArgumentError DistributedQueue(-1; spawn_worker=() -> error("should not be called"))

    # teardown_timeout=0 means "fire-and-forget, no bound" to rmprocs, not "don't wait" -- rejected along with negative/NaN.
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=() -> error("should not be called"), teardown_timeout=0)
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=() -> error("should not be called"), teardown_timeout=-1)
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=() -> error("should not be called"), teardown_timeout=NaN)

    # spawn_worker is required -- no generic implementation works for an arbitrary cluster.
    @test_throws UndefKeywordError DistributedQueue()
    @test_throws UndefKeywordError DistributedQueue(4)

    # spawn_worker just creates the process; setup_worker loads BigHO + this file's test functions onto it.
    test_spawn_worker() = first(addprocs(1))
    function test_setup_worker(pid)
        Distributed.remotecall_eval(Main, [pid], :(begin
            using BigHO
            dq_square(a) = a^2
            dq_slow(a) = (sleep(0.5); a^2)
        end))
        return nothing
    end
    @everywhere using BigHO
    @everywhere dq_square(a) = a^2
    @everywhere dq_slow(a) = (sleep(0.5); a^2)

    max_concurrency = 3
    try
        # capacity bookkeeping: consumed by submit!, restored by poll.
        ex = DistributedQueue(max_concurrency; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker)
        BigHO.start!(ex, nothing)
        @test BigHO.capacity(ex) == max_concurrency
        entry1 = BigHO.RunEntry(1, (a=1,))
        BigHO.submit!(ex, entry1, dq_square)
        @test BigHO.capacity(ex) == max_concurrency - 1
        entry2 = BigHO.RunEntry(2, (a=2,))
        BigHO.submit!(ex, entry2, dq_square)
        @test BigHO.capacity(ex) == max_concurrency - 2
        out = BigHO.poll(ex)
        append!(out, BigHO.poll(ex)) # a second call is always safe, see the analogous Threaded test
        @test length(out) == 2
        @test BigHO.capacity(ex) == max_concurrency
        @test Set(e.id for (e, _) in out) == Set([1, 2])
        BigHO.shutdown!(ex)

        # Regression: poll() must not discard earlier good results just because a later item is an interrupt.
        ex_batch = DistributedQueue(max_concurrency; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker)
        BigHO.start!(ex_batch, nothing)
        entryA = BigHO.RunEntry(1, (a=1,))
        entryB = BigHO.RunEntry(2, (a=2,))
        entryC = BigHO.RunEntry(3, (a=3,))
        put!(ex_batch.results, (entryA, BigHO.ObjectiveOutcome(10, nothing)))
        put!(ex_batch.results, (entryB, InterruptException()))
        put!(ex_batch.results, (entryC, BigHO.ObjectiveOutcome(30, nothing)))
        ex_batch.in_flight = 3
        out_batch = BigHO.poll(ex_batch)
        @test Set(e.id for (e, _) in out_batch) == Set([1, 3])
        @test BigHO.capacity(ex_batch) == 0
        @test_throws InterruptException BigHO.poll(ex_batch)
        BigHO.shutdown!(ex_batch)

        # Regression: shutdown! kills unfinished trials, but an already-completed trial's result must survive.
        shutdown_spawned = Int[]
        function shutdown_spawn_worker()
            pid = first(addprocs(1))
            push!(shutdown_spawned, pid)
            return pid
        end
        shutdown_setup_worker(pid) = Distributed.remotecall_eval(Main, [pid], :(using BigHO))
        ex_shutdown = DistributedQueue(2; spawn_worker=shutdown_spawn_worker, setup_worker=shutdown_setup_worker)
        BigHO.start!(ex_shutdown, nothing)
        BigHO.submit!(ex_shutdown, BigHO.RunEntry(1, (a=1,)), a -> a^2) # fast
        BigHO.submit!(ex_shutdown, BigHO.RunEntry(2, (a=2,)), a -> (sleep(30.0); a^2)) # slow
        sleep(5.0) # generous margin for the fast trial's own spawn+compute+report+teardown to genuinely finish
        elapsed = @elapsed BigHO.shutdown!(ex_shutdown)
        @test elapsed < 10.0 # killed the slow trial rather than waiting out its full 30s sleep
        out_shutdown = BigHO.poll(ex_shutdown)
        out_shutdown_by_id = Dict(e.id => outcome for (e, outcome) in out_shutdown)
        @test out_shutdown_by_id[1].value == 1 # the fast trial's real result, preserved
        @test out_shutdown_by_id[2] isa Exception # the slow trial, killed by shutdown!, reported failed not silently lost
        @test length(shutdown_spawned) == 2
        @test isempty(intersect(shutdown_spawned, workers())) # both torn down -- none leaked

        # shutdown! doesn't swallow a failed task's exception -- wait() raises it, same as any other error.
        ex_task_failure = DistributedQueue(1; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker)
        BigHO.start!(ex_task_failure, nothing)
        failed_task = @async throw(InterruptException())
        try
            wait(failed_task)
        catch
        end
        @test istaskfailed(failed_task)
        # Placeholder pid -- never used, since this task is already done.
        push!(ex_task_failure.tasks, (0, failed_task))
        @test_throws TaskFailedException BigHO.shutdown!(ex_task_failure)

        # Regression: an interrupt in the LOCAL supervisory task (blocked in fetch()) must be forwarded, not lost.
        # setup_done flags when setup_worker() has returned, so the interrupt reliably lands inside fetch().
        setup_done = Ref(false)
        function interrupt_setup_worker(pid)
            test_setup_worker(pid)
            setup_done[] = true
        end
        ex_interrupt = DistributedQueue(1; spawn_worker=test_spawn_worker, setup_worker=interrupt_setup_worker)
        BigHO.start!(ex_interrupt, nothing)
        BigHO.submit!(ex_interrupt, BigHO.RunEntry(1, (a=1,)), dq_slow)
        let waited = 0.0
            while !setup_done[] && waited < 60.0
                sleep(0.1)
                waited += 0.1
            end
        end
        @test setup_done[] # sanity: actually reached fetch(), not still in setup_worker
        sleep(0.1) # remotecall itself is near-instant once setup_worker() returns
        schedule(ex_interrupt.tasks[1][2], InterruptException(); error=true)
        @test isempty(BigHO.poll(ex_interrupt)) # nothing lost, just nothing to report yet
        @test_throws InterruptException BigHO.poll(ex_interrupt) # deferred throw on the next call
        BigHO.shutdown!(ex_interrupt)

        # An InterruptException from the OBJECTIVE (remote) arrives wrapped in a RemoteException, unlike a local one -- treated as an ordinary Failed trial.
        ex_objective_interrupt = DistributedQueue(1; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker)
        BigHO.start!(ex_objective_interrupt, nothing)
        BigHO.submit!(ex_objective_interrupt, BigHO.RunEntry(1, (a=1,)), a -> throw(InterruptException()))
        out_objective_interrupt = BigHO.poll(ex_objective_interrupt)
        @test length(out_objective_interrupt) == 1
        @test out_objective_interrupt[1][2] isa Exception # an ordinary Failed outcome, not an abort
        BigHO.shutdown!(ex_objective_interrupt)

        # Correctness: a full run through Hyperoptimizer/run! works, not just the low-level interface above.
        ho = Hyperoptimizer(dq_square, (a=Ordinal(1:5),); n=20)
        run!(ho; executor=DistributedQueue(max_concurrency; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker))
        @test length(results(ho)) == 20
        @test all(e -> e.status == BigHO.Completed, ho.runs)
        @test minimum(ho) == 1

        # A worker dying mid-trial reports that trial Failed without crashing the run or affecting siblings.
        # Rigged by VALUE (one param calls exit()), not by timing, which flaked on a loaded CI runner.
        n_death_trials = 4
        dying_id = 2
        death_spawned = Int[]
        function death_spawn_worker()
            pid = first(addprocs(1))
            push!(death_spawned, pid)
            return pid
        end
        death_setup_worker(pid) = Distributed.remotecall_eval(Main, [pid], :(using BigHO))
        dq_death_or_square = a -> a == dying_id ? exit() : a^2
        ex_death = DistributedQueue(n_death_trials; spawn_worker=death_spawn_worker, setup_worker=death_setup_worker)
        BigHO.start!(ex_death, nothing)
        for i in 1:n_death_trials
            BigHO.submit!(ex_death, BigHO.RunEntry(i, (a=i,)), dq_death_or_square)
        end
        # Watched with a timeout so a "worker dies -> hang" regression fails loudly instead of hanging the suite.
        collected = Tuple{BigHO.RunEntry,Any}[]
        collect_task = @async begin
            while length(collected) < n_death_trials
                append!(collected, BigHO.poll(ex_death))
            end
        end
        let waited = 0.0
            while !istaskdone(collect_task) && waited < 60.0
                sleep(0.5)
                waited += 0.5
            end
        end
        @test istaskdone(collect_task)
        @test length(collected) == n_death_trials
        @test length(death_spawned) == n_death_trials
        @test length(unique(death_spawned)) == n_death_trials # no reuse: one distinct worker per trial
        outcome_by_id = Dict(e.id => outcome for (e, outcome) in collected)
        @test outcome_by_id[dying_id] isa Exception # the trial whose worker died is reported failed, not lost
        for i in 1:n_death_trials
            i == dying_id && continue
            @test outcome_by_id[i].value == i^2 # siblings unaffected, each with its own correct result
        end
        BigHO.shutdown!(ex_death)

        # Regression: an interrupt with MULTIPLE trials in flight must abandon EVERY Pending entry (not just one) and tear down every worker.
        # SequentialSampler makes tasks[1] deterministically trial 1; waits for all 3 workers to spawn before interrupting.
        interrupt_others_spawned = Int[]
        function interrupt_others_spawn_worker()
            pid = first(addprocs(1))
            push!(interrupt_others_spawned, pid)
            return pid
        end
        interrupt_others_setup_worker(pid) = Distributed.remotecall_eval(Main, [pid], :(using BigHO))
        dq_interrupt_others = a -> (sleep(30.0); a^2) # generous margin -- Windows CI's slower process spawn eats into it
        ex_interrupt_others = DistributedQueue(3; spawn_worker=interrupt_others_spawn_worker, setup_worker=interrupt_others_setup_worker)
        ho_dq_interrupt = Hyperoptimizer(dq_interrupt_others, (a=Nominal([1, 2, 3]),);
                                          sampler=SequentialSampler(), n=3)
        run_task = @async run!(ho_dq_interrupt; executor=ex_interrupt_others)
        let waited = 0.0
            while length(interrupt_others_spawned) < 3 && waited < 60.0
                sleep(0.1)
                waited += 0.1
            end
        end
        @test length(interrupt_others_spawned) == 3 # sanity: all 3 workers spawned before we interrupt
        schedule(ex_interrupt_others.tasks[1][2], InterruptException(); error=true)
        caught = nothing
        try
            wait(run_task)
        catch e
            caught = e
        end
        @test caught isa TaskFailedException
        @test caught.task.result isa InterruptException
        @test ho_dq_interrupt.status == BigHO.Errored
        @test ho_dq_interrupt.n_pending == 0
        @test length(ho_dq_interrupt.runs) == 3
        @test count(e -> e.status == BigHO.Abandoned, ho_dq_interrupt.runs) == 3 # all 3, not just the one that threw
        @test length(interrupt_others_spawned) == 3 # one dedicated worker per trial, as always
        @test isempty(intersect(interrupt_others_spawned, workers())) # every one of them was torn down -- none leaked

        # Regression: setup_worker() throwing after spawn_worker() succeeded is an ordinary Failed trial and doesn't leak the worker.
        spawn_fail_spawned = Int[]
        function spawn_fail_spawn_worker()
            pid = first(addprocs(1))
            push!(spawn_fail_spawned, pid)
            return pid
        end
        spawn_fail_setup_worker(pid) = error("simulated cluster/remotecall_eval setup failure")
        ex_spawn_fail = DistributedQueue(1; spawn_worker=spawn_fail_spawn_worker, setup_worker=spawn_fail_setup_worker)
        BigHO.start!(ex_spawn_fail, nothing)
        BigHO.submit!(ex_spawn_fail, BigHO.RunEntry(1, (a=1,)), dq_square)
        out_spawn_fail = BigHO.poll(ex_spawn_fail)
        @test length(out_spawn_fail) == 1
        @test out_spawn_fail[1][2] isa Exception # an ordinary Failed outcome, not a hang or crash
        BigHO.shutdown!(ex_spawn_fail)
        @test length(spawn_fail_spawned) == 1
        @test isempty(intersect(spawn_fail_spawned, workers())) # setup_worker failing doesn't leak it anymore
    finally
        rmprocs(filter(!=(1), workers()))
    end
end
