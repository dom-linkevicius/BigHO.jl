@testset "Serial executor" begin
    @info "Testing Serial executor"

    # submit! runs the objective synchronously, so a result is always ready
    # immediately -- unlike Threaded, nothing is ever actually "in flight"
    # from Serial's perspective.
    ex = Serial()
    BigHO.start!(ex, nothing)
    @test BigHO.capacity(ex) == 1
    entry = BigHO.RunEntry(1, (a=1,))
    BigHO.submit!(ex, entry, a -> a)
    @test BigHO.capacity(ex) == 0 # one already-completed result waiting to be polled
    out = BigHO.poll(ex)
    @test length(out) == 1
    @test out[1][1].id == 1
    @test BigHO.capacity(ex) == 1 # draining the buffer frees capacity back up
    BigHO.shutdown!(ex)

    # An InterruptException (e.g. Ctrl+C while a trial is running) must
    # propagate out of safe_call rather than being caught and recorded as a
    # failed trial -- run! stops gracefully (ho.status = Interrupted) instead
    # of continuing as if nothing happened. The interrupted trial itself is
    # abandoned (never told an outcome), not left permanently Pending.
    let n_calls = Ref(0)
        global interrupt_after_first(a) = (n_calls[] += 1; n_calls[] == 1 ? a : throw(InterruptException()))
    end
    ho_interrupt = Hyperoptimizer(interrupt_after_first, (a=Nominal([1, 2, 3]),); n=3)
    @test_logs (:info, r"Aborting") run!(ho_interrupt)
    @test ho_interrupt.status == BigHO.Interrupted
    @test length(results(ho_interrupt)) == 1  # the trial before the interrupt completed normally
    @test ho_interrupt.n_pending == 0
    @test ho_interrupt.runs[2].status == BigHO.Abandoned # the interrupted trial itself
    @test_throws ArgumentError run!(ho_interrupt) # an Interrupted optimizer can never be resumed
    @test_throws ArgumentError settarget!(ho_interrupt, 10) # nor can its target be raised to feign otherwise

    # Regression: ask!/tell! (the manual API) must refuse a terminal
    # Hyperoptimizer too -- otherwise they're a backdoor around the
    # "Interrupted/Errored is permanent" guarantee run!/settarget! enforce,
    # letting new trials get silently asked and told on an optimizer that's
    # supposed to be done.
    @test_throws ArgumentError BigHO.ask!(ho_interrupt)
    @test_throws ArgumentError BigHO.tell!(ho_interrupt, ho_interrupt.runs[2], 42) # even for its own abandoned entry

    # Correctness: enough trials relative to the grid size (~500x oversampling
    # per candidate) to find the true optimum with overwhelming probability
    # regardless of RNG state, without depending on exact draw-position luck.
    g(a, b) = (a - 7)^2 + (b - 3)^2
    ho_exact = Hyperoptimizer(g, (a=Ordinal(0:10), b=Ordinal(0:10)); n=6000)
    run!(ho_exact; executor=Serial())
    @test minimum(ho_exact) == 0
    @test minimizer(ho_exact) == [7, 3]
end
