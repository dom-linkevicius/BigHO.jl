@testset "Executor equivalence" begin
    @info "Testing executor equivalence"

    # Executor-agnostic correctness: the SAME setup, run through Serial,
    # Threaded, and DistributedQueue, must reach the identical final result.
    # All three draw identically because RandomSampler()'s default
    # constructor builds a fresh StableRNG(1) each time it's called
    # (src/samplers/random.jl) -- not because of any shared/global RNG
    # state. ask!'s draw sequence never depends on which executor evaluates
    # the objective, and update_best!'s comparison-based tracking is
    # order-independent, so only the *order* trials complete in should ever
    # differ between executors -- never the final outcome. history/results
    # are compared as sorted multisets for exactly that reason.
    #
    # n is small (not the 500+ used to stress-test Serial/Threaded
    # elsewhere): DistributedQueue pays a real process-spawn cost (seconds)
    # per trial -- negligible for this library's actual hours-long trials,
    # but not something a unit test should multiply by hundreds of trials.
    @everywhere using BigHO
    @everywhere dq_eq(a, b) = (a - 7)^2 + (b - 3)^2
    @everywhere dq_eq_h(a) = a == 5 ? error("boom") : a

    # A real spawn_worker must set up whatever a fresh worker needs -- here,
    # BigHO plus this file's test functions -- since a freshly addprocs'd
    # worker starts with none of that loaded. remotecall_eval targets only
    # the new pid, unlike @everywhere, which would (harmlessly, but
    # wastefully) redefine these on every worker again each time a fresh one
    # is spawned.
    function test_spawn_worker()
        pid = first(addprocs(1))
        Distributed.remotecall_eval(Main, [pid], :(begin
            using BigHO
            dq_eq(a, b) = (a - 7)^2 + (b - 3)^2
            dq_eq_h(a) = a == 5 ? error("boom") : a
        end))
        return pid
    end

    try
        eq_domains = (a=Ordinal(0:10), b=Ordinal(0:10))
        eq_n = 20

        ho_s = Hyperoptimizer(dq_eq, eq_domains; n=eq_n)
        run!(ho_s; executor=Serial())
        ho_t = Hyperoptimizer(dq_eq, eq_domains; n=eq_n)
        run!(ho_t; executor=Threaded(8))
        ho_d = Hyperoptimizer(dq_eq, eq_domains; n=eq_n)
        run!(ho_d; executor=DistributedQueue(3; spawn_worker=test_spawn_worker))

        @test sort(history(ho_s)) == sort(history(ho_t)) == sort(history(ho_d))
        @test sort(results(ho_s)) == sort(results(ho_t)) == sort(results(ho_d))
        @test minimum(ho_s) == minimum(ho_t) == minimum(ho_d)
        @test minimizer(ho_s) == minimizer(ho_t) == minimizer(ho_d)

        # Same, but with a mix of Completed and Failed outcomes -- the
        # happy-path equivalence above never fails a trial, so it can't catch
        # a future race that misattributes a Failed outcome to the wrong
        # entry, or vice versa, under concurrent completion timing.
        h_domain = (a=Ordinal(1:10),)
        h_n = 20

        ho_s_mixed = Hyperoptimizer(dq_eq_h, h_domain; n=h_n)
        run!(ho_s_mixed; executor=Serial())
        ho_t_mixed = Hyperoptimizer(dq_eq_h, h_domain; n=h_n)
        run!(ho_t_mixed; executor=Threaded(8))
        ho_d_mixed = Hyperoptimizer(dq_eq_h, h_domain; n=h_n)
        run!(ho_d_mixed; executor=DistributedQueue(3; spawn_worker=test_spawn_worker))

        outcome_by_params(ho) = Dict(e.params => e.status for e in ho.runs)
        @test outcome_by_params(ho_s_mixed) == outcome_by_params(ho_t_mixed) == outcome_by_params(ho_d_mixed)
        @test BigHO.Failed in values(outcome_by_params(ho_s_mixed)) # sanity: the mix actually includes a failure
    finally
        rmprocs(filter(!=(1), workers()))
    end
end
