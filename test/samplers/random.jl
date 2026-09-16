@testset "Random sampler" begin
    @info "Testing Random sampler"
    Random.seed!(0)
    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

    hor = Hyperoptimizer(p -> f(p.a, p.b, c=p.c),
                          (a=Continuous(1, 5),
                           b=Nominal([true, false]),
                           c=Continuous(-1, 3; transform=exp10));
                          n=100)
    run!(hor)
    show(devnull, hor)
    @test minimum(hor) < 300
    @test length(history(hor)) == 100
    @test length(results(hor)) == 100
    @test all(h -> 1.0 <= h.a <= 5.0 && h.b isa Bool && 0.1 <= h.c <= 1000.0, history(hor))

    printmin(hor)

    @test_logs (:info, r"target set to 102") settarget!(hor, 102)
    run!(hor)
    @test length(history(hor)) == 102
    @test length(results(hor)) == 102

    @test_throws ArgumentError settarget!(hor, 50)

    ho_pending = Hyperoptimizer(p -> p.a, (a=Nominal([1, 2, 3]),); n=3)
    BigHO.ask!(ho_pending)
    @test_logs (:warn, r"pending") (:info, r"target set to 5") settarget!(ho_pending, 5)

    ho3 = Hyperoptimizer(p -> p.a * p.b, (a=Nominal([20]), b=Nominal([10])); n=1)
    run!(ho3)
    io = IOBuffer()
    printmin(io, ho3)
    @test String(take!(io)) == "a = 20\nb = 10\n"

    @test_throws ArgumentError Hyperoptimizer(p -> p.a, (a=[1, 2, 3],); n=3)

    @test_throws ArgumentError Hyperoptimizer(p -> p.a, (a=Nominal([1, 2, 3]),); n=-1)
    @test_throws ArgumentError Hyperoptimizer(p -> p.a, (a=Nominal([1, 2, 3]),); n=0)
end

@testset "RandomSampler interface" begin
    @info "Testing RandomSampler's exhausted/blocked always return false"
    ho = Hyperoptimizer(p -> p.a, (a=Nominal([1, 2, 3]),); n=3)
    @test !BigHO.exhausted(ho.sampler, ho)
    @test !BigHO.blocked(ho.sampler, ho)
end

@testset "Categorical" begin
    @info "Testing Categorical"
    f(a, b=true; c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

    hor = Hyperoptimizer(p -> f(p.a, p.b, c=p.c) + p.d(p.a),
                          (a=Continuous(1, 5),
                           b=Nominal([true, false]),
                           c=Continuous(-1, 3; transform=exp10),
                           d=Nominal([tanh, exp]));
                          n=100)
    run!(hor)
    show(devnull, hor)
    @test length(history(hor)) == 100
    @test length(results(hor)) == 100
    @test all(h -> h.b isa Bool && h.d in (tanh, exp), history(hor))
end

@testset "Optimization correctness and efficiency" begin
    @info "Testing that random search actually finds good/correct optima"

    f_disc(x, y) = (x - 23)^2 + (y - 17)^2
    true_min = minimum(f_disc(x, y) for x in 1:30, y in 1:30)
    @test true_min == 0
    disc_domains = (x=Ordinal(1:30), y=Ordinal(1:30))
    ho_exact = Hyperoptimizer(p -> f_disc(p.x, p.y), disc_domains; n=15000)
    run!(ho_exact)
    @test minimum(ho_exact) == true_min
    @test minimizer(ho_exact) == [23, 17]

    ho_small = Hyperoptimizer(p -> f_disc(p.x, p.y), disc_domains; n=10)
    run!(ho_small)
    @test minimum(ho_small) > minimum(ho_exact)

    target = Float64(pi)
    ho_cont = Hyperoptimizer(p -> (p.x - target)^2, (x=Continuous(-10, 10),); n=3000)
    run!(ho_cont)
    @test abs(minimizer(ho_cont)[1] - target) < 0.05

    target2 = (2.71828, -4.5)
    ho_cont2d = Hyperoptimizer(p -> (p.a - target2[1])^2 + (p.b - target2[2])^2,
                                (a=Continuous(-10, 10), b=Continuous(-10, 10)); n=5000)
    run!(ho_cont2d)
    m = minimizer(ho_cont2d)
    @test abs(m[1] - target2[1]) < 0.5
    @test abs(m[2] - target2[2]) < 0.5
end

@testset "Non-numerics" begin
    @info "Testing optimizing over non-numeric elements"
    hor = Hyperoptimizer(p -> p.g(p.x), (g=Nominal([sin, exp, identity]), x=Continuous(0, 1)); n=100)
    run!(hor)
    show(devnull, hor)
    @test minimum(hor) < ℯ
    @test length(history(hor)) == 100
    @test length(results(hor)) == 100
    @test all(h -> h.g in (sin, exp, identity) && 0.0 <= h.x <= 1.0, history(hor))
end
