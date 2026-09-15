import ProgressMeter

@testset "Progress" begin
    @info "Testing Progress"

    @test_throws UndefKeywordError Hyperoptimizer(p -> p.a, (a=Nominal([1, 2, 3]),))

    ho = Hyperoptimizer(p -> p.a^2, (a=Ordinal(1:20),); n=10)
    run!(ho; executor=Serial())
    @test length(results(ho)) == 10
    @test ho.status == BigHO.Finished

    ho2 = Hyperoptimizer(p -> p.a^2, (a=Ordinal(1:20),); n=5)
    run!(ho2; executor=Serial())
    settarget!(ho2, 10)
    run!(ho2; executor=Serial())
    @test length(results(ho2)) == 10

    buf = IOBuffer()
    p = ProgressMeter.Progress(10; output=buf)
    ProgressMeter.update!(p, 5; force=true)
    for i in 6:10
        ProgressMeter.update!(p, i)
    end
    ProgressMeter.finish!(p)
    @test !isempty(String(take!(buf)))

    ho3 = Hyperoptimizer(p -> p.a^2, (a=Ordinal(1:20),); n=5)
    run!(ho3; executor=Serial(), show_progress=false)
    @test length(results(ho3)) == 5
end
