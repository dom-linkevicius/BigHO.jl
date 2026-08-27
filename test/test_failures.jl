@testset "Failure handling" begin
    @info "Testing NaN and exception failure handling"

    # NaN is excluded as a valid optimization value: a trial whose objective
    # returns NaN is marked Failed (not Completed), warns, and is excluded
    # from history/results -- it never wins minimum/minimizer.
    let n_calls = Ref(0)
        global nan_after_first(a, b) = (n_calls[] += 1; n_calls[] == 1 ? a * b : NaN)
    end
    ho_nan = Hyperoptimizer(nan_after_first, (a=Nominal([20]), b=Nominal([1])); n=2)
    @test_logs (:warn, r"NaN") run!(ho_nan)
    @test minimum(ho_nan) == 20
    @test minimizer(ho_nan) == [20, 1]
    @test length(ho_nan.runs) == 2         # both trials are recorded...
    @test length(results(ho_nan)) == 1    # ...but only the non-NaN one is Completed
    @test !any(isnan, results(ho_nan))
    @test ho_nan.runs[2].status == Hyperopt.Failed
    @test ho_nan.runs[2].value === missing

    # A thrown exception behaves the same way: Failed, warns showing what was
    # thrown, excluded from history/results.
    let n_calls = Ref(0)
        global throws_after_first(a, b) = (n_calls[] += 1; n_calls[] == 1 ? a * b : error("boom"))
    end
    ho_err = Hyperoptimizer(throws_after_first, (a=Nominal([20]), b=Nominal([1])); n=2)
    @test_logs (:warn, r"non-Real") run!(ho_err)
    @test minimum(ho_err) == 20
    @test minimizer(ho_err) == [20, 1]
    @test length(ho_err.runs) == 2
    @test length(results(ho_err)) == 1
    @test ho_err.runs[2].status == Hyperopt.Failed
    @test ho_err.runs[2].value === missing

    # A run where every trial fails has no completed trial at all -- the
    # optimum accessors throw rather than returning a sentinel.
    ho_allnan = Hyperoptimizer((a) -> NaN, (a=Nominal([1]),); n=1)
    @test_logs (:warn, r"NaN") run!(ho_allnan)
    @test length(results(ho_allnan)) == 0
    @test_throws ErrorException minimum(ho_allnan)
    @test_throws ErrorException minimizer(ho_allnan)

    # The failing trial's actual parameter values (labeled by name, not just
    # a bare tuple) must be visible in the warning, not just the failure
    # itself -- `@test_logs` above only checks the message text, so capture
    # the fully rendered log output here to confirm the params really show up.
    let n_calls = Ref(0)
        global nan_with_distinctive_params(a, b) = (n_calls[] += 1; n_calls[] == 1 ? a + b : NaN)
    end
    ho_params = Hyperoptimizer(nan_with_distinctive_params, (a=Nominal([777]), b=Nominal([888])); n=2)
    io = IOBuffer()
    Logging.with_logger(Logging.ConsoleLogger(io)) do
        run!(ho_params)
    end
    log_text = String(take!(io))
    @test occursin("777", log_text)
    @test occursin("888", log_text)
    @test occursin("a = 777", log_text) # labeled by param name, not a bare positional tuple
    @test occursin("b = 888", log_text)
end
