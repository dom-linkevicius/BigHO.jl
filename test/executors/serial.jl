@testset "Serial executor" begin
    @info "Testing Serial executor"

    ex = Serial()
    BigHO.start!(ex, nothing)
    @test BigHO.capacity(ex) == 1
    entry = BigHO.RunEntry(1, (a=1,), Float64[])
    BigHO.submit!(ex, entry, p -> p.a)
    @test BigHO.capacity(ex) == 0
    out = BigHO.poll(ex)
    @test length(out) == 1
    @test out[1][1].id == 1
    @test BigHO.capacity(ex) == 1
    BigHO.shutdown!(ex)

    let n_calls = Ref(0)
        global interrupt_after_first(p) = (n_calls[] += 1; n_calls[] == 1 ? p.a : throw(InterruptException()))
    end
    ho_interrupt = Hyperoptimizer(interrupt_after_first, (a=Nominal([1, 2, 3]),); n=3)
    @test_throws InterruptException run!(ho_interrupt)
    @test ho_interrupt.status == BigHO.Errored
    @test length(results(ho_interrupt)) == 1
    @test ho_interrupt.n_pending == 0
    @test ho_interrupt.runs[2].status == BigHO.Abandoned
    @test_throws ArgumentError run!(ho_interrupt)
    @test_throws ArgumentError settarget!(ho_interrupt, 10)

    @test_throws ArgumentError BigHO.ask!(ho_interrupt)
    @test_throws ArgumentError BigHO.tell!(ho_interrupt, ho_interrupt.runs[2], 42)

    g(p) = (p.a - 7)^2 + (p.b - 3)^2
    ho_exact = Hyperoptimizer(g, (a=Ordinal(0:10), b=Ordinal(0:10)); n=6000)
    run!(ho_exact; executor=Serial())
    @test minimum(ho_exact) == 0
    @test minimizer(ho_exact) == [7, 3]
end
