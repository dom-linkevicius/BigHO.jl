@testset "Random sampler" begin
    @info "Testing Random sampler"
    Random.seed!(0)
    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2) # must be defined outside testsets to avoid scoping issues

    hor = Hyperoptimizer((a, b, c) -> f(a, b, c=c),
                          (a=Continuous(1, 5, 4 / 49),                    # 50 evenly spaced points
                           b=Nominal([true, false]),
                           c=Ordinal(exp10.(LinRange(-1, 3, 50))));        # log-spaced -- Ordinal here works just as well as Continuous(values) would
                          n=100)
    run!(hor)
    show(devnull, hor)
    @test minimum(hor) < 300
    @test length(history(hor)) == 100
    @test length(results(hor)) == 100
    @test all(history(hor)) do h
        all(hi in hor.candidates[i] for (i, hi) in enumerate(h))
    end

    printmin(hor)

    # Resuming a run is just raising the target via `settarget!` and calling
    # `run!` again -- no special macro/capture mechanism, state already lives in `hor`.
    @test_logs (:info, r"target set to 102") settarget!(hor, 102)
    run!(hor)
    @test length(history(hor)) == 102
    @test length(results(hor)) == 102

    # settarget! can only raise the target, never lower it.
    @test_throws ArgumentError settarget!(hor, 50)

    ho3 = Hyperoptimizer((a, b) -> a * b, (a=Nominal([20]), b=Nominal([10])); n=1)
    run!(ho3)
    io = IOBuffer()
    printmin(io, ho3)
    @test String(take!(io)) == "a = 20\nb = 10\n"

    # candidates must be Domain objects -- a plain array is rejected up front.
    @test_throws ArgumentError Hyperoptimizer((a) -> a, (a=[1, 2, 3],))
end

@testset "Categorical" begin
    @info "Testing Categorical"
    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

    hor = Hyperoptimizer((a, b, c, d) -> f(a, b, c=c) + d(a),
                          (a=Continuous(1, 5, 4 / 49),
                           b=Nominal([true, false]),
                           c=Ordinal(exp10.(LinRange(-1, 3, 50))),
                           d=Nominal([tanh, exp]));
                          n=100)
    run!(hor)
    show(devnull, hor)
    @test length(history(hor)) == 100
    @test length(results(hor)) == 100
    @test all(history(hor)) do h
        all(hi in hor.candidates[i] for (i, hi) in enumerate(h))
    end
end

@testset "Non-numerics" begin
    @info "Testing optimizing over non-numeric elements"
    hor = Hyperoptimizer((g, x) -> g(x), (g=Nominal([sin, exp, identity]), x=Continuous(0, 1, 1 / 99)); n=100)
    run!(hor)
    show(devnull, hor)
    @test minimum(hor) < ℯ
    @test length(history(hor)) == 100
    @test length(results(hor)) == 100
    @test all(history(hor)) do h
        all(hi in hor.candidates[i] for (i, hi) in enumerate(h))
    end
end
