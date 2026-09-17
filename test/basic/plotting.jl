@testset "summaryplot" begin
    @info "Testing summaryplot(ho)"

    ho = Hyperoptimizer(p -> (p.a - 3)^2 + (p.b - 1)^2,
                         (a=Continuous(0, 10), b=Nominal([1, 2, 3, 4, 5])); n=20)
    run!(ho; show_progress=false)
    fig = summaryplot(ho)
    @test fig isa CairoMakie.Figure

    ho_single = Hyperoptimizer(p -> p.a^2, (a=Nominal([1, 2, 3]),); n=3)
    run!(ho_single; show_progress=false)
    @test summaryplot(ho_single) isa CairoMakie.Figure

    ho_empty = Hyperoptimizer(nothing, (a=Nominal([1]),); n=1)
    @test_throws ErrorException summaryplot(ho_empty)

    @test summaryplot(ho;
        figure_kwargs=(; size=(500, 500)),
        axis_kwargs=(; titlesize=10),
        scatter_kwargs=(; color=:red),
        histogram_kwargs=(; color=:orange),
        line_kwargs=(; color=:green)) isa CairoMakie.Figure
end

@testset "summaryplot with categorical hyperparameters" begin
    @info "Testing summaryplot(ho) when a hyperparameter's values are not numbers"

    ho = Hyperoptimizer(p -> (p.a - 3)^2 + (p.kernel == "rbf" ? 0.0 : 1.0),
                         (a=Continuous(0, 10), kernel=Nominal(["rbf", "linear"]), shape=Nominal([:wide, :narrow]));
                         n=20)
    run!(ho; show_progress=false)

    @test eltype(DataFrame(ho).kernel) <: AbstractString
    @test summaryplot(ho) isa CairoMakie.Figure
    @test summaryplot(ho; histogram_kwargs=(; color=:orange)) isa CairoMakie.Figure

    ho_only_categorical = Hyperoptimizer(p -> p.kernel == "rbf" ? 0.0 : 1.0,
                                          (kernel=Nominal(["rbf", "linear"]),); n=6)
    run!(ho_only_categorical; show_progress=false)
    @test summaryplot(ho_only_categorical) isa CairoMakie.Figure
end
