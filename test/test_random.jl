@testset "Random sampler" begin
    @info "Testing Random sampler"
    Random.seed!(0)
    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2) # must be defined outside testsets to avoid scoping issues

    hor = Hyperoptimizer((a, b, c) -> f(a, b, c=c),
                          (a=Continuous(1, 5, 4 / 49),                    # 50 evenly spaced points
                           b=Nominal([true, false]),
                           c=Continuous(exp10.(LinRange(-1, 3, 50))));
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

    # n must be non-negative -- a negative target would otherwise silently
    # "reach target" with zero completed runs and run! would do nothing,
    # with no error or warning explaining why.
    @test_throws ArgumentError Hyperoptimizer((a) -> a, (a=Nominal([1, 2, 3]),); n=-1)
    @test Hyperoptimizer((a) -> a, (a=Nominal([1, 2, 3]),); n=0) isa Hyperoptimizer
end

@testset "Interrupt handling" begin
    @info "Testing InterruptException handling"

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

@testset "Categorical" begin
    @info "Testing Categorical"
    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

    hor = Hyperoptimizer((a, b, c, d) -> f(a, b, c=c) + d(a),
                          (a=Continuous(1, 5, 4 / 49),
                           b=Nominal([true, false]),
                           c=Continuous(exp10.(LinRange(-1, 3, 50))),
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

@testset "Optimization correctness and efficiency" begin
    @info "Testing that random search actually finds good/correct optima"

    # Correctness: for a domain small enough to compute a ground truth by
    # brute force, random search given enough trials must find the TRUE
    # global optimum exactly, not just something plausible.
    f_disc(x, y) = (x - 73)^2 + (y - 91)^2
    xs, ys = collect(1:100), collect(1:100)
    true_min = minimum(f_disc(x, y) for x in xs, y in ys)
    @test true_min == 0
    ho_exact = Hyperoptimizer((x, y) -> f_disc(x, y), (x=Continuous(1, 100, 1), y=Continuous(1, 100, 1)); n=15000)
    run!(ho_exact)
    @test minimum(ho_exact) == true_min
    @test minimizer(ho_exact) == [73, 91]

    # Efficiency: more trials should never do worse, and in practice does
    # substantially better -- the same objective/domain with a much smaller
    # budget must not find as good a result as the run above.
    ho_small = Hyperoptimizer((x, y) -> f_disc(x, y), (x=Continuous(1, 100, 1), y=Continuous(1, 100, 1)); n=10)
    run!(ho_small)
    @test minimum(ho_small) > minimum(ho_exact)

    # Correctness on a continuous domain: converges close to a target that
    # isn't exactly on the discretization grid (π, not a round number), to
    # actual grid-resolution precision rather than by grid-alignment luck.
    target = Float64(pi)
    ho_cont = Hyperoptimizer(x -> (x - target)^2, (x=Continuous(-10, 10, 0.01),); n=3000)
    run!(ho_cont)
    @test abs(minimizer(ho_cont)[1] - target) < 0.01 # within one grid step

    # Same, in two dimensions -- a harder search (more candidate combinations
    # for the same per-dimension budget), so a looser but still meaningful bound.
    target2 = (2.71828, -4.5)
    ho_cont2d = Hyperoptimizer((a, b) -> (a - target2[1])^2 + (b - target2[2])^2,
                                (a=Continuous(-10, 10, 0.01), b=Continuous(-10, 10, 0.01)); n=5000)
    run!(ho_cont2d)
    m = minimizer(ho_cont2d)
    @test abs(m[1] - target2[1]) < 0.15
    @test abs(m[2] - target2[2]) < 0.15
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
