@testset "summaryplot" begin
    @info "Testing summaryplot(ho)"

    ho = Hyperoptimizer((a, b) -> (a - 3)^2 + (b - 1)^2,
                         (a=Continuous(0, 10, 0.1), b=Nominal([1, 2, 3, 4, 5])); n=20)
    run!(ho; show_progress=false)
    fig = summaryplot(ho)
    @test fig isa CairoMakie.Figure

    # A single-parameter Hyperoptimizer still works -- one marginal-scatter row.
    ho_single = Hyperoptimizer(a -> a^2, (a=Nominal([1, 2, 3]),); n=3)
    run!(ho_single; show_progress=false)
    @test summaryplot(ho_single) isa CairoMakie.Figure

    # No completed runs -- nothing to plot, so it throws rather than silently
    # producing an empty/broken figure.
    ho_empty = Hyperoptimizer(nothing, (a=Nominal([1]),); n=1)
    @test_throws ErrorException summaryplot(ho_empty)

    # Custom kwargs are forwarded to Figure/Axis/the scatter, histogram, and
    # line marks -- just check they're accepted without erroring.
    @test summaryplot(ho;
        figure_kwargs=(; size=(500, 500)),
        axis_kwargs=(; titlesize=10),
        scatter_kwargs=(; color=:red),
        histogram_kwargs=(; color=:orange),
        line_kwargs=(; color=:green)) isa CairoMakie.Figure
end
