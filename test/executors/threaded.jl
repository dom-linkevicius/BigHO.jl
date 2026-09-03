# Throws on its 3rd call, to exercise an exception from ask! rather than the objective.
mutable struct BuggySampler <: BigHO.Sampler
    n_calls::Int
end
BuggySampler() = BuggySampler(0)
function (s::BuggySampler)(ctx)
    s.n_calls += 1
    s.n_calls == 3 && error("sampler bug")
    return [s.n_calls]
end
BigHO.init(s::BuggySampler, ctx) = s
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
    # poll isn't guaranteed to drain both in one call -- call again, like run! does in a loop.
    out = BigHO.poll(ex)
    append!(out, BigHO.poll(ex))
    @test length(out) == 2
    @test BigHO.capacity(ex) == 2
    @test Set(e.id for (e, _) in out) == Set([1, 2])
    BigHO.shutdown!(ex)

    # Regression: poll() must not discard earlier good results just because a later item is an interrupt.
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

    # Regression: shutdown! must wait for outstanding tasks, not just stop tracking them.
    ex_shutdown = Threaded(4)
    BigHO.start!(ex_shutdown, nothing)
    finished = Ref(false)
    BigHO.submit!(ex_shutdown, BigHO.RunEntry(1, (a=1,)), _ -> (sleep(0.3); finished[] = true; 1))
    BigHO.shutdown!(ex_shutdown)
    @test finished[]

    # shutdown! doesn't swallow a failed task's exception -- wait() raises it, same as any other error.
    failed_task = @async throw(InterruptException())
    try
        wait(failed_task)
    catch
    end
    @test istaskfailed(failed_task)
    ex_task_failure = Threaded(1)
    BigHO.start!(ex_task_failure, nothing)
    push!(ex_task_failure.tasks, failed_task)
    @test_throws TaskFailedException BigHO.shutdown!(ex_task_failure)

    # Every trial told exactly once, correctly attributed, regardless of completion order under real concurrency.
    ho = Hyperoptimizer(a -> a^2, (a=Continuous(1, 500, 1),); n=500)
    run!(ho; executor=Threaded(8))
    @test length(ho.runs) == 500
    @test all(e -> e.status == BigHO.Completed, ho.runs) # none lost/left Pending
    @test all(e -> e.value == e.params.a^2, ho.runs) # each outcome attributed to the right entry

    # An exception thrown inside a spawned task is caught and recorded as Failed, not left to crash the run.
    ho_err = Hyperoptimizer(a -> a == 2 ? error("boom") : a, (a=Nominal([1, 2, 3]),); n=3)
    ex_err = Threaded(3)
    @test_logs (:warn, r"non-Real") run!(ho_err; executor=ex_err)
    @test length(results(ho_err)) == 2
    @test length(ho_err.runs) == 3
    @test count(e -> e.status == BigHO.Failed, ho_err.runs) == 1

    # An interrupt inside a spawned task must not deadlock poll() -- run! rethrows it with valid partial results.
    let n_calls = Ref(0)
        global interrupt_after_first_threaded(a) = (n_calls[] += 1; n_calls[] == 1 ? a : throw(InterruptException()))
    end
    ho_interrupt = Hyperoptimizer(interrupt_after_first_threaded, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws InterruptException run!(ho_interrupt; executor=Threaded(1))
    @test ho_interrupt.status == BigHO.Errored
    @test length(results(ho_interrupt)) == 1  # the trial before the interrupt completed normally
    @test ho_interrupt.n_pending == 0
    @test ho_interrupt.runs[2].status == BigHO.Abandoned # the interrupted trial itself
    @test_throws ArgumentError run!(ho_interrupt) # an Errored optimizer can never be resumed

    # Regression: a bug in the Sampler's ask! call must propagate like an interrupt, abandoning any trial still in flight.
    ho_errored = Hyperoptimizer(a -> a == 1 ? 1.0 : (sleep(2.0); 2.0), (a=Nominal([1, 2, 3]),);
                                 sampler=BuggySampler(), n=3)
    # Checks the message, not just the type -- a prior bug raised the wrong ErrorException here.
    run_errored_exception = nothing
    try
        run!(ho_errored; executor=Threaded(2))
    catch e
        run_errored_exception = e
    end
    @test run_errored_exception isa ErrorException
    @test run_errored_exception.msg == "sampler bug"
    @test ho_errored.status == BigHO.Errored
    @test ho_errored.n_pending == 0
    @test length(ho_errored.runs) == 2 # the 3rd draw's entry is never created; ask! throws before that
    @test count(e -> e.status == BigHO.Completed, ho_errored.runs) == 1 # the fast trial
    @test count(e -> e.status == BigHO.Abandoned, ho_errored.runs) == 1 # the still-sleeping sibling
    @test_throws ArgumentError run!(ho_errored) # an Errored optimizer can never be resumed either
    @test_throws ArgumentError settarget!(ho_errored, 10)
    @test_throws ArgumentError BigHO.ask!(ho_errored) # nor can the manual ask!/tell! API sneak past this
    @test_throws ArgumentError BigHO.tell!(ho_errored, ho_errored.runs[2], 42) # even for its own abandoned entry

    # Correctness: enough oversampling to find the true optimum regardless of RNG state.
    g(a, b) = (a - 7)^2 + (b - 3)^2
    ho_exact = Hyperoptimizer(g, (a=Ordinal(0:10), b=Ordinal(0:10)); n=6000)
    run!(ho_exact; executor=Threaded(8))
    @test minimum(ho_exact) == 0
    @test minimizer(ho_exact) == [7, 3]

    # Concurrency actually happens: delayed trials finish well under serial time.
    # Thread count asserted directly, since sleep's cooperative yielding could pass this on 1 thread too.
    @test Threads.nthreads() > 1
    n_slow, delay = 8, 0.2
    ho_slow = Hyperoptimizer(_ -> (sleep(delay); 1.0), (a=Nominal(1:n_slow),); n=n_slow)
    elapsed = @elapsed run!(ho_slow; executor=Threaded(n_slow))
    @test elapsed < n_slow * delay / 2 # well under fully-serial time, generous margin against CI jitter
end
