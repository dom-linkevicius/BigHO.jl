@testset "Random sampler" begin
    @info "Testing Random sampler"
    Random.seed!(0)
    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2) # must be defined outside testsets to avoid scoping issues

    hor = Hyperoptimizer((a, b, c) -> f(a, b, c=c),
                          (a=Continuous(1, 5, 4 / 49),                    # 50 evenly spaced points
                           b=Levels([true, false]),
                           c=Levels(exp10.(LinRange(-1, 3, 50))));        # log-spaced -> non-uniform grid, so Levels not Continuous
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

    # Test NaN handling: NaN is a legitimate objective value, distinct from a failure.
    let n_calls = Ref(0)
        global nan_after_first(a, b) = (n_calls[] += 1; n_calls[] == 1 ? a * b : NaN)
    end
    ho2 = Hyperoptimizer(nan_after_first, (a=Levels([20]), b=Levels([1])); n=2)
    run!(ho2)
    @test minimum(ho2) == 20
    @test minimizer(ho2) == [20, 1]
    @test length(results(ho2)) == 2 # the legitimate NaN is recorded, not dropped
    @test any(isnan, results(ho2))

    ho3 = Hyperoptimizer((a, b) -> a * b, (a=Levels([20]), b=Levels([10])); n=1)
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
                           b=Levels([true, false]),
                           c=Levels(exp10.(LinRange(-1, 3, 50))),
                           d=Categorical([tanh, exp]));
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
    hor = Hyperoptimizer((g, x) -> g(x), (g=Categorical([sin, exp, identity]), x=Continuous(0, 1, 1 / 99)); n=100)
    run!(hor)
    show(devnull, hor)
    @test minimum(hor) < ℯ
    @test length(history(hor)) == 100
    @test length(results(hor)) == 100
    @test all(history(hor)) do h
        all(hi in hor.candidates[i] for (i, hi) in enumerate(h))
    end
end
