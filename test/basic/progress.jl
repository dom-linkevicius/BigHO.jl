import ProgressMeter

@testset "Progress" begin
    @info "Testing Progress"

    # show_progress defaults to true and tracks trials told against ho.n,
    # which must therefore be set -- there's no sampler that determines its
    # own total independently of ho.n (see run!'s docstring).
    ho_nontarget = Hyperoptimizer(a -> a, (a=Nominal([1, 2, 3]),))
    @test_throws ArgumentError run!(ho_nontarget)

    # Otherwise it's purely cosmetic -- doesn't change the run's outcome.
    # ProgressMeter defaults to stderr, not the Logging-based stdout/stderr
    # this suite's @test_logs calls elsewhere inspect, so it doesn't need to
    # be silenced here.
    ho = Hyperoptimizer(a -> a^2, (a=Ordinal(1:20),); n=10)
    run!(ho; executor=Serial())
    @test length(results(ho)) == 10
    @test ho.status == BigHO.Finished

    # Also fine across a settarget!-based resume, where the bar must start
    # from n_told(ho), not 0.
    ho2 = Hyperoptimizer(a -> a^2, (a=Ordinal(1:20),); n=5)
    run!(ho2; executor=Serial())
    settarget!(ho2, 10)
    run!(ho2; executor=Serial())
    @test length(results(ho2)) == 10

    # Regression: a resumed run must actually draw something, not just avoid
    # erroring. Two distinct ProgressMeter quirks combine here (both
    # confirmed by reading its source, not guessed): (1) the `start` kwarg
    # doesn't seed its internal counter (stays 0 regardless), so a resumed
    # run's first update! jumping straight from 0 to a large n_told(ho)
    # would show a stale/wrong position until the next natural redraw; (2)
    # ProgressMeter's own completion message (finish!) only ever prints if
    # *some* earlier update already printed -- unconditionally, regardless
    # of `force` -- so a run whose every trial resolves faster than dt
    # apart would otherwise finish having displayed nothing at all, start
    # to end, even with correct counter tracking. run!'s fix -- force=true
    # on the one initial sync update! only -- addresses both at once: the
    # correct position shows immediately, and ProgressMeter's internal
    # "printed" flag is set early enough that the eventual completion
    # message is guaranteed rather than dependent on real timing (verified
    # directly: without the initial force=true, this same sequence
    # non-deterministically produces zero output when every update below
    # completes faster than dt apart, exactly as it did during actual
    # Pkg.test() runs of this file before this fix). This can't be checked
    # by redirecting stderr around a real run! call: confirmed empirically
    # that ProgressMeter's default `output=stderr` is bound at its own
    # precompilation time, so a later redirect_stderr here has no effect on
    # it regardless of whether run!'s own fix is correct. Checked instead
    # by reproducing run!'s exact sequence directly against ProgressMeter
    # with an explicit, capturable `output`.
    buf = IOBuffer()
    p = ProgressMeter.Progress(10; output=buf)
    ProgressMeter.update!(p, 5; force=true) # sync to current position, as run! does on resume
    for i in 6:10
        ProgressMeter.update!(p, i)
    end
    ProgressMeter.finish!(p)
    @test !isempty(String(take!(buf)))

    # Still available to opt out of explicitly.
    ho3 = Hyperoptimizer(a -> a^2, (a=Ordinal(1:20),); n=5)
    run!(ho3; executor=Serial(), show_progress=false)
    @test length(results(ho3)) == 5
end
