@testset "Executor equivalence" begin
    @info "Testing executor equivalence"

    # The SAME setup, run through all three executors, must reach the identical final result.
    # n is small: DistributedQueue pays a real process-spawn cost per trial.
    @everywhere using BigHO
    @everywhere dq_eq(a, b) = (a - 7)^2 + (b - 3)^2
    @everywhere dq_eq_h(a) = a == 5 ? error("boom") : a

    # spawn_worker just creates the process; setup_worker loads BigHO + this file's test functions onto it.
    test_spawn_worker() = first(addprocs(1))
    function test_setup_worker(pid)
        Distributed.remotecall_eval(Main, [pid], :(begin
            using BigHO
            dq_eq(a, b) = (a - 7)^2 + (b - 3)^2
            dq_eq_h(a) = a == 5 ? error("boom") : a
        end))
        return nothing
    end

    try
        eq_domains = (a=Ordinal(0:10), b=Ordinal(0:10))
        eq_n = 20

        # Threaded(8) only proves equivalence under real concurrency if there's more than one thread.
        @test Threads.nthreads() > 1

        ho_s = Hyperoptimizer(dq_eq, eq_domains; n=eq_n)
        run!(ho_s; executor=Serial())
        ho_t = Hyperoptimizer(dq_eq, eq_domains; n=eq_n)
        run!(ho_t; executor=Threaded(8))
        ho_d = Hyperoptimizer(dq_eq, eq_domains; n=eq_n)
        run!(ho_d; executor=DistributedQueue(3; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker))

        @test sort(history(ho_s)) == sort(history(ho_t)) == sort(history(ho_d))
        @test sort(results(ho_s)) == sort(results(ho_t)) == sort(results(ho_d))
        @test minimum(ho_s) == minimum(ho_t) == minimum(ho_d)
        @test minimizer(ho_s) == minimizer(ho_t) == minimizer(ho_d)

        # Same, but with a mix of Completed/Failed outcomes, to catch a race that misattributes a Failed outcome.
        h_domain = (a=Ordinal(1:5),)
        h_n = 20

        ho_s_mixed = Hyperoptimizer(dq_eq_h, h_domain; n=h_n)
        run!(ho_s_mixed; executor=Serial())
        ho_t_mixed = Hyperoptimizer(dq_eq_h, h_domain; n=h_n)
        run!(ho_t_mixed; executor=Threaded(8))
        ho_d_mixed = Hyperoptimizer(dq_eq_h, h_domain; n=h_n)
        run!(ho_d_mixed; executor=DistributedQueue(3; spawn_worker=test_spawn_worker, setup_worker=test_setup_worker))

        outcome_by_params(ho) = Dict(e.params => e.status for e in ho.runs)
        @test outcome_by_params(ho_s_mixed) == outcome_by_params(ho_t_mixed) == outcome_by_params(ho_d_mixed)
        @test BigHO.Failed in values(outcome_by_params(ho_s_mixed)) # sanity: the mix actually includes a failure
    finally
        rmprocs(filter(!=(1), workers()))
    end
end
