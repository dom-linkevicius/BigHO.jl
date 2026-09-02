import ProgressMeter

@testset "Progress" begin
    @info "Testing Progress"

    # show_progress defaults to true and requires ho.n to be set (see run!'s docstring).
    ho_nontarget = Hyperoptimizer(a -> a, (a=Nominal([1, 2, 3]),))
    @test_throws ArgumentError run!(ho_nontarget)

    # Purely cosmetic -- doesn't change the outcome. ProgressMeter writes directly to stderr, not through Logging, so it doesn't need silencing here.
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

    # Regression: force=true fixes a resumed bar silently printing nothing (stale start position + finish! no-ops unless something printed first).
    # Reproduces run!'s update sequence directly via output=buf, since ProgressMeter's stderr default is bound at precompile time -- redirect_stderr wouldn't catch it here.
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
