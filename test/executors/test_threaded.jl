@testset "Threaded executor" begin
    @info "Testing Threaded executor"

    @test_throws ArgumentError Threaded(0)
    @test_throws ArgumentError Threaded(-1)

    # capacity bookkeeping: consumed by submit!, restored by poll.
    ex = Threaded(2)
    Hyperopt.start!(ex, nothing)
    @test Hyperopt.capacity(ex) == 2
    entry1 = Hyperopt.RunEntry(1, (a=1,))
    Hyperopt.submit!(ex, entry1, a -> a)
    @test Hyperopt.capacity(ex) == 1
    entry2 = Hyperopt.RunEntry(2, (a=2,))
    Hyperopt.submit!(ex, entry2, a -> a)
    @test Hyperopt.capacity(ex) == 0
    # poll blocks for the first result but isn't guaranteed to drain both in
    # one call (the second trivial task may not have finished yet under
    # system load) -- call again rather than assume a single call suffices,
    # matching how run! itself actually consumes poll() (in a loop). A second
    # call is always safe: once in_flight hits 0 it returns empty immediately
    # rather than blocking.
    out = Hyperopt.poll(ex)
    append!(out, Hyperopt.poll(ex))
    @test length(out) == 2
    @test Hyperopt.capacity(ex) == 2
    @test Set(e.id for (e, _) in out) == Set([1, 2])
    Hyperopt.shutdown!(ex)

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
    @test all(e -> e.status == Hyperopt.Completed, ho.runs) # none lost/left Pending
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
    @test count(e -> e.status == Hyperopt.Failed, ho_err.runs) == 1

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
