# A minimal custom Sampler producing candidates 1, 2, 3, ... in that exact
# order every time, regardless of concurrency downstream -- ask! itself is
# always called sequentially from run!'s own driving loop (protected by
# ho.lock), even though what happens to each drawn value afterward (on the
# executor side) can run concurrently. Lets one specific trial be rigged to
# behave differently by VALUE rather than by luck of a random draw or by
# timing -- the same principle death_spawn_worker below already relies on
# via directly-constructed RunEntry ids, just usable here at the ask!/run!
# level instead of the low-level submit! API.
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

    # teardown_timeout=0 isn't "don't wait" to Distributed.rmprocs -- it's a
    # different, fire-and-forget code path with no bound at all, silently
    # reinstating the exact unbounded worker_lock hold teardown_timeout
    # exists to prevent. Negative/NaN are equally nonsensical. All three must
    # be rejected up front rather than accepted and misbehave later.
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=() -> error("should not be called"), teardown_timeout=0)
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=() -> error("should not be called"), teardown_timeout=-1)
    @test_throws ArgumentError DistributedQueue(1; spawn_worker=() -> error("should not be called"), teardown_timeout=NaN)

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

        # Regression: shutdown! must also recognize an interrupt that fails a
        # TRIAL's own task (as opposed to landing directly on shutdown!'s own
        # wait(t) call above) -- the case added in round 3 for an interrupt
        # landing inside _teardown_worker, after that trial's outcome has
        # already been put! and reported. wait() on a task that itself failed
        # wraps the exception in TaskFailedException too, which is exactly
        # what this elseif branch exists to unwrap; confirmed (per the round-4
        # review that flagged this branch as untested) that deleting it still
        # passes the whole suite, so this pins the behavior directly. Seeded
        # by hand, like ex_batch above, since real timing can't reliably land
        # an interrupt inside this exact window.
        ex_task_interrupt = DistributedQueue(1; spawn_worker=test_spawn_worker)
        BigHO.start!(ex_task_interrupt, nothing)
        failed_task = @async throw(InterruptException())
        try
            wait(failed_task)
        catch
        end
        @test istaskfailed(failed_task)
        push!(ex_task_interrupt.tasks, failed_task)
        @test_throws InterruptException BigHO.shutdown!(ex_task_interrupt)

        # Regression: a genuine interrupt landing inside the LOCAL supervisory
        # task (blocked in fetch(), not the remote objective throwing -- see
        # the dedicated test for that below, since it's a genuinely different
        # code path) must not deadlock or get lost -- forwarded through the
        # channel and deferred to the next poll() call, exactly like Threaded.
        # Injected via schedule(...; error=true), the same
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

        # Regression: an InterruptException thrown by the OBJECTIVE itself
        # runs on a remote worker, so it crosses back to the driver through
        # fetch() wrapped in a RemoteException (confirmed empirically --
        # fetch() on a future whose remote task threw InterruptException
        # raises RemoteException(pid, CapturedException(InterruptException(),
        # ...)), not the raw InterruptException the fetch()-side test above
        # covers), and safe_call's own contract is that InterruptException is
        # the one exception that must never be silently recorded as an
        # ordinary Failed outcome the way any other remote error correctly
        # would be.
        ex_objective_interrupt = DistributedQueue(1; spawn_worker=test_spawn_worker)
        BigHO.start!(ex_objective_interrupt, nothing)
        BigHO.submit!(ex_objective_interrupt, BigHO.RunEntry(1, (a=1,)), a -> throw(InterruptException()))
        @test isempty(BigHO.poll(ex_objective_interrupt)) # nothing lost, just nothing to report yet
        @test_throws InterruptException BigHO.poll(ex_objective_interrupt) # deferred throw on the next call
        BigHO.shutdown!(ex_objective_interrupt)

        # Correctness: a full run through Hyperoptimizer/run! actually works
        # and finds a sane answer, not just the low-level interface above.
        # Domain kept small (not the 500+ oversampling used to stress-test
        # Serial/Threaded elsewhere) rather than raising n to get the same
        # margin -- each trial here pays a real process-spawn cost, so
        # growing n would meaningfully slow this test; n=20 over 5 values is
        # still a healthy ~4x oversampling instead of a bare 1x.
        ho = Hyperoptimizer(dq_square, (a=Ordinal(1:5),); n=20)
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
        # The dying trial is chosen by VALUE, not by timing: since each
        # trial gets its own dedicated worker (never shared), rigging one
        # specific parameter value to call exit() is deterministic --
        # there's no race to decide which worker to kill or when, unlike
        # guessing from wall-clock timing (which flaked on a loaded CI
        # runner -- see git history). fetch() on that one trial's Future
        # throws ProcessExitedException exactly as it would for a real
        # crash, without disturbing its siblings, each computing on their
        # own separate, unaffected process.
        n_death_trials = 4
        dying_id = 2
        death_spawned = Int[]
        function death_spawn_worker()
            pid = first(addprocs(1))
            push!(death_spawned, pid)
            Distributed.remotecall_eval(Main, [pid], :(using BigHO))
            return pid
        end
        dq_death_or_square = a -> a == dying_id ? exit() : a^2
        ex_death = DistributedQueue(n_death_trials; spawn_worker=death_spawn_worker)
        BigHO.start!(ex_death, nothing)
        for i in 1:n_death_trials
            BigHO.submit!(ex_death, BigHO.RunEntry(i, (a=i,)), dq_death_or_square)
        end
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
        outcome_by_id = Dict(e.id => outcome for (e, outcome) in collected)
        @test outcome_by_id[dying_id] isa Exception # the trial whose worker died is reported failed, not lost
        for i in 1:n_death_trials
            i == dying_id && continue
            @test outcome_by_id[i].value == i^2 # siblings unaffected, each with its own correct result
        end
        BigHO.shutdown!(ex_death)

        # Regression: an interrupt reaching Hyperoptimizer/run! through
        # DistributedQueue, with MULTIPLE other trials genuinely in flight
        # (not just one), must correctly (a) mark ho.status Interrupted, (b)
        # reclassify EVERY still-Pending entry to Abandoned -- not just the
        # first one found, which is exactly what an accidental early `break`
        # in that loop would still get away with if only ever tested at
        # concurrency 1 (as every other executor's interrupt test in this
        # suite is) -- and (c) actually tear down every one of those other
        # in-flight workers once shutdown! (in run!'s own finally) finishes
        # waiting them out, leaking none of them.
        #
        # Rigged by VALUE via SequentialSampler, not by wall-clock timing:
        # trial 1 always throws immediately (once its own worker is ready);
        # trials 2-3 always sleep far longer than trial 1's own
        # spawn+dispatch+detect overhead could plausibly take, so they're
        # provably still genuinely Pending -- not just recently finished --
        # at the moment _abandon_pending! runs.
        interrupt_others_spawned = Int[]
        function interrupt_others_spawn_worker()
            pid = first(addprocs(1))
            push!(interrupt_others_spawned, pid)
            Distributed.remotecall_eval(Main, [pid], :(using BigHO))
            return pid
        end
        dq_interrupt_others = a -> a == 1 ? throw(InterruptException()) : (sleep(8.0); a^2)
        ho_dq_interrupt = Hyperoptimizer(dq_interrupt_others, (a=Nominal([1, 2, 3]),);
                                          sampler=SequentialSampler(), n=3)
        @test_logs (:info, r"Aborting") run!(ho_dq_interrupt; executor=DistributedQueue(3; spawn_worker=interrupt_others_spawn_worker))
        @test ho_dq_interrupt.status == BigHO.Interrupted
        @test ho_dq_interrupt.n_pending == 0
        @test length(ho_dq_interrupt.runs) == 3
        @test count(e -> e.status == BigHO.Abandoned, ho_dq_interrupt.runs) == 3 # all 3, not just the one that threw
        @test length(interrupt_others_spawned) == 3 # one dedicated worker per trial, as always
        @test isempty(intersect(interrupt_others_spawned, workers())) # every one of them was torn down -- none leaked

        # Regression: spawn_worker() itself throwing a non-interrupt exception
        # AFTER already provisioning a worker (e.g. addprocs succeeded but a
        # later setup step like remotecall_eval failed) must not crash or hang
        # submit!/poll -- it's absorbed as an ordinary Failed trial, same as
        # any other per-trial failure, since BigHO is never given a pid to
        # tear down when spawn_worker never returns one (see the module
        # docstring's note on spawn_worker's own cleanup responsibility in
        # that case). Tracks the pid it creates, like death_spawn_worker
        # above, just to confirm a worker really was provisioned before the
        # induced failure -- this testset's own outer rmprocs cleanup below
        # reaps it, since BigHO itself has no pid to do so with.
        spawn_fail_spawned = Int[]
        function spawn_fail_spawn_worker()
            pid = first(addprocs(1))
            push!(spawn_fail_spawned, pid)
            error("simulated cluster/remotecall_eval setup failure")
        end
        ex_spawn_fail = DistributedQueue(1; spawn_worker=spawn_fail_spawn_worker)
        BigHO.start!(ex_spawn_fail, nothing)
        BigHO.submit!(ex_spawn_fail, BigHO.RunEntry(1, (a=1,)), dq_square)
        out_spawn_fail = BigHO.poll(ex_spawn_fail)
        @test length(out_spawn_fail) == 1
        @test out_spawn_fail[1][2] isa Exception # an ordinary Failed outcome, not a hang or crash
        BigHO.shutdown!(ex_spawn_fail)
        @test length(spawn_fail_spawned) == 1 # confirms this exercised "worker created, then setup failed"
    finally
        rmprocs(filter(!=(1), workers()))
    end
end
