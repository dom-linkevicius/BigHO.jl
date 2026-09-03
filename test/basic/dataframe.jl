@testset "DataFrame" begin
    @info "Testing DataFrame(ho)"

    ho = Hyperoptimizer(a -> a^2, (a=Nominal([1, 2, 3]),); n=3)
    run!(ho; show_progress=false)
    df = DataFrame(ho)
    @test size(df) == (3, 7)
    @test names(df) == ["id", "status", "a", "value", "error", "pre_artefact", "post_artefact"]
    @test df.id == [1, 2, 3]
    @test all(df.status .== BigHO.Completed)
    @test all(df.value .== df.a .^ 2)
    @test all(df.error .=== nothing)
    @test all(df.pre_artefact .=== nothing)
    @test all(df.post_artefact .=== nothing)

    # A Failed trial still gets a row -- status distinguishes it, value is missing,
    # and the raw error is preserved -- rather than being silently dropped like
    # history/results.
    let n_calls = Ref(0)
        global dataframe_nan_after_first(a) = (n_calls[] += 1; n_calls[] == 1 ? a : NaN)
    end
    ho_nan = Hyperoptimizer(dataframe_nan_after_first, (a=Nominal([1, 2]),); n=2)
    @test_logs (:warn, r"NaN") run!(ho_nan; show_progress=false)
    df_nan = DataFrame(ho_nan)
    @test size(df_nan) == (2, 7)
    @test count(==(BigHO.Failed), df_nan.status) == 1
    @test ismissing(df_nan.value[df_nan.status.==BigHO.Failed][1])
    @test isnan(df_nan.error[df_nan.status.==BigHO.Failed][1])

    # A Stateful objective's pre_artefact/post_artefact land in their own columns.
    ho_stateful = Hyperoptimizer(Stateful((a; pre_artefact=nothing) -> (a, "artefact-$a")), (a=Nominal([1, 2]),); n=2)
    run!(ho_stateful; show_progress=false)
    df_stateful = DataFrame(ho_stateful)
    @test df_stateful.post_artefact == ["artefact-$a" for a in df_stateful.a]

    # Multiple hyperparameters each get their own column, named after the param.
    ho_multi = Hyperoptimizer((a, b) -> a + b, (a=Nominal([1]), b=Nominal([2])); n=1)
    run!(ho_multi; show_progress=false)
    @test names(DataFrame(ho_multi)) == ["id", "status", "a", "b", "value", "error", "pre_artefact", "post_artefact"]

    # No trials yet -- still produces a DataFrame, just with zero rows.
    ho_empty = Hyperoptimizer(nothing, (a=Nominal([1]),); n=1)
    @test size(DataFrame(ho_empty)) == (0, 0)
end
