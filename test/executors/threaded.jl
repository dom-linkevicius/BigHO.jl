# A minimal custom Sampler whose (sampler)(ctx) call itself throws on its
# 3rd invocation, regardless of what the objective does -- used to exercise
# an exception escaping run!'s own orchestration (ask!) rather than the
# objective (which safe_call already handles separately).
mutable struct BuggySampler <: BigHO.Sampler
    n_calls::Int
end
BuggySampler() = BuggySampler(0)
function (s::BuggySampler)(ctx)
    s.n_calls += 1
    s.n_calls == 3 && error("sampler bug")
    return [s.n_calls]
end
BigHO.init!(::BuggySampler, ctx) = nothing
BigHO.exhausted(::BuggySampler, ho) = false
BigHO.on_tell!(::BuggySampler, entry) = nothing

@testset "Threaded executor" begin
    @info "Testing Threaded executor"

    @test_throws ArgumentError Threaded(0)
    @test_throws ArgumentError Threaded(-1)

    # capacity bookkeeping: consumed by submit!, restored by poll.
    ex = Threaded(2)
    BigHO.start!(ex, nothing)
    @test BigHO.capacity(ex) == 2
    entry1 = BigHO.RunEntry(1, (a=1,))
    BigHO.submit!(ex, entry1, a -> a)
    @test BigHO.capacity(ex) == 1
    entry2 = BigHO.RunEntry(2, (a=2,))
    BigHO.submit!(ex, entry2, a -> a)
    @test BigHO.capacity(ex) == 0
    # poll blocks for the first result but isn't guaranteed to drain both in
    # one call (the second trivial task may not have finished yet under
    # system load) -- call again rather than assume a single call suffices,
    # matching how run! itself actually consumes poll() (in a loop). A second
    # call is always safe: once in_flight hits 0 it returns empty immediately
    # rather than blocking.
    out = BigHO.poll(ex)
    append!(out, BigHO.poll(ex))
    @test length(out) == 2
    @test BigHO.capacity(ex) == 2
    @test Set(e.id for (e, _) in out) == Set([1, 2])
    BigHO.shutdown!(ex)

    # Regression: poll() must not discard already-collected results from
    # earlier in the same batch just because a later item in that batch is an
    # interrupt -- it defers the throw to the NEXT call instead, so
    # already-completed trials are never silently lost. Seeded directly
    # (rather than via run!) since real concurrency timing can't
    # deterministically force multiple results and an interrupt into the same
    # poll() batch.
    ex_batch = Threaded(4)
    BigHO.start!(ex_batch, nothing)
    entryA = BigHO.RunEntry(1, (a=1,))
    entryB = BigHO.RunEntry(2, (a=2,))
    entryC = BigHO.RunEntry(3, (a=3,))
    put!(ex_batch.results, (entryA, BigHO.ObjectiveOutcome(10, nothing)))
    put!(ex_batch.results, (entryB, InterruptException()))
    put!(ex_batch.results, (entryC, BigHO.ObjectiveOutcome(30, nothing)))
    ex_batch.in_flight = 3
    out_batch = BigHO.poll(ex_batch)
    @test Set(e.id for (e, _) in out_batch) == Set([1, 3]) # both good results preserved, not discarded
    @test BigHO.capacity(ex_batch) == 0 # a pending exception blocks further submission immediately
    @test_throws InterruptException BigHO.poll(ex_batch) # deferred throw happens on the NEXT call
    BigHO.shutdown!(ex_batch)

    # Regression: shutdown! must actually wait for outstanding tasks to
    # finish, not just stop tracking them -- otherwise a trial still running
    # when the run ends keeps executing invisibly in the background after the
    # caller already got control back. Tested at the executor level directly
    # (rather than via run!) for a deterministic outcome regardless of
    # RandomSampler's draw order.
    ex_shutdown = Threaded(4)
    BigHO.start!(ex_shutdown, nothing)
    finished = Ref(false)
    BigHO.submit!(ex_shutdown, BigHO.RunEntry(1, (a=1,)), _ -> (sleep(0.3); finished[] = true; 1))
    BigHO.shutdown!(ex_shutdown)
    @test finished[]

    # Regression: shutdown! must let a genuine InterruptException propagate
    # (e.g. a second Ctrl+C while already waiting out a task that's still
    # running) rather than silently swallowing it -- only non-interrupt
    # exceptions are this function's business to absorb. wait() on a failed
    # task wraps the real exception in a TaskFailedException, so check
    # .task.result for the actual InterruptException underneath.
    ex_shutdown_interrupt = Threaded(1)
    BigHO.start!(ex_shutdown_interrupt, nothing)
    BigHO.submit!(ex_shutdown_interrupt, BigHO.RunEntry(1, (a=1,)), _ -> (sleep(2.0); 1))
    shutdown_task = @async BigHO.shutdown!(ex_shutdown_interrupt)
    sleep(0.2)
    schedule(shutdown_task, InterruptException(); error=true)
    caught = nothing
    try
        wait(shutdown_task)
    catch e
        caught = e
    end
    @test caught isa TaskFailedException
    @test caught.task.result isa InterruptException

    # Every trial gets told exactly once, with the outcome correctly
    # attributed to its own entry, regardless of completion order under real
    # concurrency -- the thing that's actually at risk in a threaded executor
    # (lost/duplicated/misattributed results), rather than in the ask!/tell!
    # core itself, which is executor-agnostic. (RandomSampler draws with
    # replacement, so n=500 over 500 candidates does NOT guarantee every
    # candidate gets visited -- that's not what's being tested here.)
    ho = Hyperoptimizer(a -> a^2, (a=Continuous(1, 500, 1),); n=500)
    run!(ho; executor=Threaded(8))
    @test length(ho.runs) == 500
    @test all(e -> e.status == BigHO.Completed, ho.runs) # none lost/left Pending
    @test all(e -> e.value == e.params.a^2, ho.runs) # each outcome attributed to the right entry

    # Failure handling goes through the same safe_call/finalize_entry pipeline
    # regardless of executor -- an exception thrown inside a spawned task is
    # still caught and recorded as Failed, not left to crash the run. The
    # executor is constructed outside @test_logs so its own "single thread"
    # warning (unrelated to this test) isn't part of the expected log sequence.
    ho_err = Hyperoptimizer(a -> a == 2 ? error("boom") : a, (a=Nominal([1, 2, 3]),); n=3)
    ex_err = Threaded(3)
    @test_logs (:warn, r"non-Real") run!(ho_err; executor=ex_err)
    @test length(results(ho_err)) == 2
    @test length(ho_err.runs) == 3
    @test count(e -> e.status == BigHO.Failed, ho_err.runs) == 1

    # Graceful interrupt under real concurrency: an InterruptException thrown
    # inside a spawned task must not deadlock poll() waiting forever for a
    # result that task can now never produce -- run! stops gracefully
    # (ho.status = Interrupted) with valid partial results, same contract as
    # Serial, and the interrupted trial itself is abandoned rather than left
    # permanently Pending. max_concurrency=1 keeps this deterministic (one
    # trial in flight at a time, so the second call is unambiguously the one
    # that interrupts).
    let n_calls = Ref(0)
        global interrupt_after_first_threaded(a) = (n_calls[] += 1; n_calls[] == 1 ? a : throw(InterruptException()))
    end
    ho_interrupt = Hyperoptimizer(interrupt_after_first_threaded, (a=Nominal([1, 2, 3]),); n=3)
    @test_logs (:info, r"Aborting") run!(ho_interrupt; executor=Threaded(1))
    @test ho_interrupt.status == BigHO.Interrupted
    @test length(results(ho_interrupt)) == 1  # the trial before the interrupt completed normally
    @test ho_interrupt.n_pending == 0
    @test ho_interrupt.runs[2].status == BigHO.Abandoned # the interrupted trial itself
    @test_throws ArgumentError run!(ho_interrupt) # an Interrupted optimizer can never be resumed

    # Regression: an exception OTHER than InterruptException escaping run!'s
    # own orchestration (here, a bug in a custom Sampler's ask!-time call,
    # not the objective -- an objective's own exception never reaches this
    # far, since safe_call already catches and records it as Failed) must
    # still propagate to the caller, unlike InterruptException, which run!
    # absorbs and reports gracefully -- silently swallowing a genuine bug
    # would make it far harder to notice or debug. ho still stops and
    # abandons any trial genuinely still in flight, so a caller who catches
    # this and retries doesn't busy-loop on stale state (the bug this whole
    # OptimizerStatus mechanism exists to prevent). max_concurrency=2, with
    # the first trial resolving near-instantly and the second sleeping much
    # longer, deterministically keeps the second trial genuinely Pending at
    # the exact moment the sampler throws on its 3rd call.
    ho_errored = Hyperoptimizer(a -> a == 1 ? 1.0 : (sleep(2.0); 2.0), (a=Nominal([1, 2, 3]),);
                                 sampler=BuggySampler(), n=3)
    @test_throws ErrorException run!(ho_errored; executor=Threaded(2))
    @test ho_errored.status == BigHO.Errored
    @test ho_errored.n_pending == 0
    @test length(ho_errored.runs) == 2 # the 3rd draw's entry is never created; ask! throws before that
    @test count(e -> e.status == BigHO.Completed, ho_errored.runs) == 1 # the fast trial
    @test count(e -> e.status == BigHO.Abandoned, ho_errored.runs) == 1 # the still-sleeping sibling
    @test_throws ArgumentError run!(ho_errored) # an Errored optimizer can never be resumed either
    @test_throws ArgumentError settarget!(ho_errored, 10)
    @test_throws ErrorException BigHO.ask!(ho_errored) # nor can the manual ask!/tell! API sneak past this
    @test_throws ErrorException BigHO.tell!(ho_errored, ho_errored.runs[2], 42) # even for its own abandoned entry

    # Correctness: enough trials relative to the grid size (~500x oversampling
    # per candidate) to find the true optimum with overwhelming probability
    # regardless of RNG state, without depending on exact draw-position luck.
    g(a, b) = (a - 7)^2 + (b - 3)^2
    ho_exact = Hyperoptimizer(g, (a=Ordinal(0:10), b=Ordinal(0:10)); n=6000)
    run!(ho_exact; executor=Threaded(8))
    @test minimum(ho_exact) == 0
    @test minimizer(ho_exact) == [7, 3]

    # Concurrency actually happens: trials with a real (I/O-bound) delay
    # complete well under their combined serial duration when run through
    # Threaded, since each is spawned as its own task rather than awaited one
    # at a time -- true even on a single OS thread, since `sleep`
    # cooperatively yields. That cooperative-yielding behavior means the
    # timing check below would still pass under a single thread without ever
    # having exercised real Base.Threads parallelism -- the actual point of
    # this phase (see the plan). Assert the thread count directly so that
    # case fails loudly instead of this concern silently "passing" without
    # having proven anything about true parallelism.
    @test Threads.nthreads() > 1
    n_slow, delay = 8, 0.2
    ho_slow = Hyperoptimizer(_ -> (sleep(delay); 1.0), (a=Nominal(1:n_slow),); n=n_slow)
    elapsed = @elapsed run!(ho_slow; executor=Threaded(n_slow))
    @test elapsed < n_slow * delay / 2 # well under fully-serial time, generous margin against CI jitter
end
