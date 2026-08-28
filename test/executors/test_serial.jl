@testset "Serial executor" begin
    @info "Testing Serial executor"

    # submit! runs the objective synchronously, so a result is always ready
    # immediately -- unlike Threaded, nothing is ever actually "in flight"
    # from Serial's perspective.
    ex = Serial()
    Hyperopt.start!(ex, nothing)
    @test Hyperopt.capacity(ex) == 1
    entry = Hyperopt.RunEntry(1, (a=1,))
    Hyperopt.submit!(ex, entry, a -> a)
    @test Hyperopt.capacity(ex) == 0 # one already-completed result waiting to be polled
    out = Hyperopt.poll(ex)
    @test length(out) == 1
    @test out[1][1].id == 1
    @test Hyperopt.capacity(ex) == 1 # draining the buffer frees capacity back up
    Hyperopt.shutdown!(ex)

    # An InterruptException (e.g. Ctrl+C while a trial is running) must
    # propagate out of safe_call rather than being caught and recorded as a
    # failed trial -- run! stops gracefully (ho.done = true) instead of
    # continuing as if nothing happened.
    let n_calls = Ref(0)
        global interrupt_after_first(a) = (n_calls[] += 1; n_calls[] == 1 ? a : throw(InterruptException()))
    end
    ho_interrupt = Hyperoptimizer(interrupt_after_first, (a=Nominal([1, 2, 3]),); n=3)
    @test_logs (:info, r"Aborting") run!(ho_interrupt)
    @test ho_interrupt.done
    @test length(results(ho_interrupt)) == 1  # the trial before the interrupt completed normally
    @test ho_interrupt.n_pending == 1         # the interrupted trial itself is never told, stays Pending
end
