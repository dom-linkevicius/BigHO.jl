@testset "DataFrame" begin
    @info "Testing DataFrame(ho)"

    ho = Hyperoptimizer(a -> a^2, (a=Nominal([1, 2, 3]),); n=3)
    run!(ho; show_progress=false)
    df = DataFrame(ho)
    @test size(df) == (3, 4)
    @test names(df) == ["id", "status", "a", "value"]
    @test df.id == [1, 2, 3]
    @test all(df.status .== BigHO.Completed)
    @test all(df.value .== df.a .^ 2)

    # A Failed trial still gets a row -- status distinguishes it, value is missing --
    # rather than being silently dropped like history/results.
    let n_calls = Ref(0)
        global dataframe_nan_after_first(a) = (n_calls[] += 1; n_calls[] == 1 ? a : NaN)
    end
    ho_nan = Hyperoptimizer(dataframe_nan_after_first, (a=Nominal([1, 2]),); n=2)
    @test_logs (:warn, r"NaN") run!(ho_nan; show_progress=false)
    df_nan = DataFrame(ho_nan)
    @test size(df_nan) == (2, 4)
    @test count(==(BigHO.Failed), df_nan.status) == 1
    @test ismissing(df_nan.value[df_nan.status.==BigHO.Failed][1])

    # Multiple hyperparameters each get their own column, named after the param.
    ho_multi = Hyperoptimizer((a, b) -> a + b, (a=Nominal([1]), b=Nominal([2])); n=1)
    run!(ho_multi; show_progress=false)
    @test names(DataFrame(ho_multi)) == ["id", "status", "a", "b", "value"]

    # No trials yet -- still produces a DataFrame, just with zero rows.
    ho_empty = Hyperoptimizer(nothing, (a=Nominal([1]),); n=1)
    @test size(DataFrame(ho_empty)) == (0, 0)
end
