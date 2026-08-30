@testset "DistributedQueue executor" begin
    @info "Testing DistributedQueue executor"

    @test_throws ArgumentError DistributedQueue(0; spawn_worker=() -> error("should not be called"))
    @test_throws ArgumentError DistributedQueue(-1; spawn_worker=() -> error("should not be called"))

    # spawn_worker has no default -- there's no generic implementation that
    # could work for an arbitrary objective (see the docstring) -- so it's a
    # required keyword argument, and omitting it fails loudly and immediately
    # rather than silently marking every trial Failed later.
    @test_throws UndefKeywordError DistributedQueue()
    @test_throws UndefKeywordError DistributedQueue(4)

    # A real usage of spawn_worker must set up whatever the new worker needs
    # to run trials -- here, BigHO itself plus this file's test functions --
    # since a freshly addprocs'd worker starts with none of that loaded.
    # remotecall_eval targets only the new pid, unlike @everywhere, which
    # would (harmlessly, but wastefully) redefine these on every worker again
    # each time a fresh one is spawned.
    function test_spawn_worker()
        pid = first(addprocs(1))
        Distributed.remotecall_eval(Main, [pid], :(begin
            using BigHO
            dq_square(a) = a^2
            dq_slow(a) = (sleep(0.5); a^2)
        end))
        return pid
    end
    @everywhere using BigHO
    @everywhere dq_square(a) = a^2
    @everywhere dq_slow(a) = (sleep(0.5); a^2)
    @everywhere dq_death_slow(a) = (sleep(8.0); a^2)

    max_concurrency = 3
    try
        # capacity bookkeeping: consumed by submit!, restored by poll. Each
        # submit! spawns (and, once resolved, tears down) its own worker --
        # there's no persistent pool field to inspect here, unlike Threaded.
        ex = DistributedQueue(max_concurrency; spawn_worker=test_spawn_worker)
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

        # Regression: poll() must not discard already-collected results from
        # earlier in the same batch just because a later item in that batch is
        # an interrupt -- see the identical Threaded regression test for the
        # full rationale. Seeded directly since real timing can't
        # deterministically force this exact interleaving.
        ex_batch = DistributedQueue(max_concurrency; spawn_worker=test_spawn_worker)
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

        # Regression: shutdown! must actually wait for outstanding tasks (here,
        # the local supervisory task fetching a remote Future, then tearing
        # down its worker) to finish. Verified via timing, not a shared Ref
        # set by the closure -- unlike Threaded, the closure here runs on a
        # REMOTE worker, which only ever mutates its own deserialized copy of
        # a captured Ref, never the original in this process.
        ex_shutdown = DistributedQueue(1; spawn_worker=test_spawn_worker)
        BigHO.start!(ex_shutdown, nothing)
        BigHO.submit!(ex_shutdown, BigHO.RunEntry(1, (a=1,)), a -> (sleep(0.3); a))
        elapsed = @elapsed BigHO.shutdown!(ex_shutdown)
        @test elapsed > 0.25 # shutdown! actually waited, rather than returning immediately

        # Regression: shutdown! must let a genuine InterruptException
        # propagate (e.g. a second Ctrl+C while already waiting out a
        # stuck/slow worker) rather than silently swallowing it. wait() on a
        # failed task wraps the real exception in a TaskFailedException, so
        # check .task.result for the actual InterruptException underneath.
        # A few seconds' delay before interrupting -- not exact timing like
        # the fetch()-vs-spawn() distinction elsewhere -- just needs
        # shutdown!'s wait(t) to still be genuinely blocked when it fires,
        # which holds throughout spawning AND the 5s simulated trial below.
        ex_shutdown_interrupt = DistributedQueue(1; spawn_worker=test_spawn_worker)
        BigHO.start!(ex_shutdown_interrupt, nothing)
        BigHO.submit!(ex_shutdown_interrupt, BigHO.RunEntry(1, (a=1,)), a -> (sleep(5.0); a))
        shutdown_task = @async BigHO.shutdown!(ex_shutdown_interrupt)
        sleep(3.0)
        schedule(shutdown_task, InterruptException(); error=true)
        caught = nothing
        try
            wait(shutdown_task)
        catch e
            caught = e
        end
        @test caught isa TaskFailedException
        @test caught.task.result isa InterruptException

        # Regression: a genuine interrupt landing inside the LOCAL supervisory
        # task (blocked in fetch(), not the remote objective throwing -- that
        # would come back wrapped in a RemoteException instead, since it
        # happens on the worker) must not deadlock or get lost -- forwarded
        # through the channel and deferred to the next poll() call, exactly
        # like Threaded. Injected via schedule(...; error=true), the same
        # primitive Julia's own runtime uses to deliver a real Ctrl+C, since
        # there's no reliable way to force a real interrupt at a precise
        # moment in a test.
        #
        # A fixed short sleep can't reliably land the interrupt inside
        # fetch(): spawn_worker() (addprocs + remotecall_eval) alone takes
        # multi-second real time, so a naive sleep(0.2) here would actually
        # catch the task mid-spawn, before remotecall/fetch are ever reached
        # -- spawn_worker itself flags when it has returned, and only once
        # that's confirmed (polled with a timeout, not guessed) do we know
        # dispatch has happened and the task is now blocked in fetch().
        spawned = Ref(false)
        function interrupt_spawn_worker()
            pid = test_spawn_worker()
            spawned[] = true
            return pid
        end
        ex_interrupt = DistributedQueue(1; spawn_worker=interrupt_spawn_worker)
        BigHO.start!(ex_interrupt, nothing)
        BigHO.submit!(ex_interrupt, BigHO.RunEntry(1, (a=1,)), dq_slow)
        let waited = 0.0
            while !spawned[] && waited < 60.0
                sleep(0.1)
                waited += 0.1
            end
        end
        @test spawned[] # sanity: actually reached fetch(), not still spawning
        sleep(0.1) # remotecall itself is near-instant once spawn_worker() returns
        schedule(ex_interrupt.tasks[1], InterruptException(); error=true)
        @test isempty(BigHO.poll(ex_interrupt)) # nothing lost, just nothing to report yet
        @test_throws InterruptException BigHO.poll(ex_interrupt) # deferred throw on the next call
        BigHO.shutdown!(ex_interrupt)

        # Correctness: a full run through Hyperoptimizer/run! actually works
        # and finds a sane answer, not just the low-level interface above.
        ho = Hyperoptimizer(dq_square, (a=Ordinal(1:20),); n=20)
        run!(ho; executor=DistributedQueue(max_concurrency; spawn_worker=test_spawn_worker))
        @test length(results(ho)) == 20
        @test all(e -> e.status == BigHO.Completed, ho.runs)
        @test minimum(ho) == 1

        # The centerpiece of this phase: a worker dying mid-trial must not
        # crash the run or corrupt state -- that trial is reported Failed,
        # everything else completes normally, and run! finishes rather than
        # aborting. Also confirms the "no reuse" guarantee directly: exactly
        # one freshly spawned, never-repeated worker per trial.
        #
        # Timing here has to respect addprocs' real cost (empirically ~2-6s
        # each on this machine, not milliseconds), which is why: (a) waiting
        # for spawns to register polls with a generous timeout instead of a
        # short fixed sleep, and (b) the simulated trial itself sleeps far
        # longer than that worst-case spawn variance, so a worker that
        # spawned quickly hasn't already finished and torn itself down by
        # the time a slower sibling is still spawning.
        n_death_trials = 4
        death_spawned = Int[]
        function death_spawn_worker()
            pid = first(addprocs(1))
            push!(death_spawned, pid)
            Distributed.remotecall_eval(Main, [pid], :(using BigHO; dq_death_slow(a) = (sleep(8.0); a^2)))
            return pid
        end
        ex_death = DistributedQueue(n_death_trials; spawn_worker=death_spawn_worker)
        BigHO.start!(ex_death, nothing)
        for i in 1:n_death_trials
            BigHO.submit!(ex_death, BigHO.RunEntry(i, (a=i,)), dq_death_slow)
        end
        # Kill the FIRST worker to actually spawn, as soon as possible --
        # waiting for every worker to spawn before killing one risks the
        # earliest spawner having already finished its whole
        # spawn-run-self-teardown cycle by the time the slowest sibling's
        # addprocs finally resolves, since spawning several workers
        # concurrently can itself take much longer than any one trial's
        # simulated 8s if e.g. precompilation contends across them.
        let waited = 0.0
            while isempty(death_spawned) && waited < 60.0
                sleep(0.1)
                waited += 0.1
            end
        end
        @test !isempty(death_spawned)
        rmprocs(death_spawned[1])
        # Collected via its own task and watched with a timeout, rather than
        # a bare while-loop, so a future regression that turns "worker dies
        # -> exception" into "worker dies -> hang" fails this test loudly
        # instead of hanging the whole suite -- and critically, the
        # surrounding `finally`'s rmprocs cleanup still runs right after
        # (killing worker processes doesn't require this task to ever
        # unblock), so nothing leaks even if it does time out.
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
        n_ok = count(pair -> !(pair[2] isa Exception), collected)
        n_failed = count(pair -> pair[2] isa Exception, collected)
        @test n_ok >= 1     # trials on their own healthy workers completed normally
        @test n_failed >= 1 # the trial whose worker died is reported failed, not lost
        @test n_ok + n_failed == n_death_trials
        BigHO.shutdown!(ex_death)
    finally
        rmprocs(filter(!=(1), workers()))
    end
end
