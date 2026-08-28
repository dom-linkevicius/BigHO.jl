mutable struct FakeNetwork
    weight::Float64
end

function train_step(lr, momentum; pre_artefact=nothing)
    net = pre_artefact === nothing ? FakeNetwork(0.0) : pre_artefact
    net.weight += lr * momentum
    loss = abs(net.weight - 1.0)
    return loss, net # (metric, post_artefact)
end

struct LoggingWrapper{F}
    f::F
    log::Vector{Any}
end
BigHO.call_objective(w::LoggingWrapper, params, pre_artefact) = (push!(w.log, pre_artefact); w.f(params...))

@testset "Stateful objectives / artefacts" begin
    @info "Testing Stateful objective artefact threading"

    # Plain (non-Stateful) objectives: pre_artefact/post_artefact are always
    # nothing -- the common case, requiring no special handling anywhere.
    ho_plain = Hyperoptimizer((a, b) -> a + b, (a=Nominal([1, 2]), b=Nominal([3, 4])); n=3)
    run!(ho_plain)
    @test all(e -> e.pre_artefact === nothing, ho_plain.runs)
    @test all(e -> e.post_artefact === nothing, ho_plain.runs)

    # Stateful: the objective receives pre_artefact (nothing here, since
    # RandomSampler never sets one) and returns (metric, post_artefact),
    # which land on .value/.post_artefact respectively -- ranking/minimum/
    # minimizer operate on the scalar metric alone, never on the artefact.
    ho = Hyperoptimizer(Stateful(train_step), (lr=Nominal([0.1, 0.5, 1.0]), momentum=Nominal([1.0])); n=3)
    run!(ho)
    @test all(v -> v isa Float64, results(ho))               # value is the scalar metric, not a tuple
    @test all(e -> e.pre_artefact === nothing, ho.runs)       # RandomSampler never sets pre_artefact (yet)
    @test all(e -> e.post_artefact isa FakeNetwork, ho.runs)  # but post_artefact is always populated
    @test minimum(ho) isa Float64
    @test ho.runs[ho.best_min_id].post_artefact isa FakeNetwork # the winning trial's artefact is retrievable

    # A Stateful objective's failure modes (NaN metric, thrown exception) are
    # handled identically to a plain objective's -- via the same finalize_entry
    # dispatch, since by the time it's caught the outcome is just a plain
    # value or exception either way.
    ho_nan = Hyperoptimizer(Stateful((a; pre_artefact=nothing) -> (NaN, "unused")), (a=Nominal([1]),); n=1)
    @test_logs (:warn, r"NaN") run!(ho_nan)
    @test length(results(ho_nan)) == 0

    ho_err = Hyperoptimizer(Stateful((a; pre_artefact=nothing) -> error("boom")), (a=Nominal([1]),); n=1)
    @test_logs (:warn, r"non-Real") run!(ho_err)
    @test length(results(ho_err)) == 0

    # A user can define their own wrapper + call_objective method for custom
    # artefact-handling behavior beyond Stateful's opaque passthrough --
    # confirms the extension point is genuinely open, not special-cased to
    # Stateful internally.
    log = Any[]
    ho_custom = Hyperoptimizer(LoggingWrapper((a, b) -> a + b, log), (a=Nominal([1]), b=Nominal([2])); n=1)
    run!(ho_custom)
    @test log == [nothing]
    @test minimum(ho_custom) == 3

    # Regression: a PLAIN (non-Stateful) objective returning its own 2-tuple
    # (loss, diagnostic_info) is never misinterpreted as a (metric,
    # post_artefact) pair just because it happens to structurally look like
    # one -- dispatch is on the dedicated ObjectiveOutcome wrapper
    # call_objective produces, never on the shape of what the objective
    # actually returned. Since only a Real is a valid Completed outcome,
    # this non-Real tuple is correctly marked Failed instead, the same as
    # any other non-Real return -- not silently split, coerced, or ranked.
    ho_tuple = Hyperoptimizer((a, b) -> (a + b, "diagnostic"), (a=Nominal([1]), b=Nominal([2])); n=1)
    @test_logs (:warn, r"non-Real") run!(ho_tuple)
    @test length(results(ho_tuple)) == 0
    @test ho_tuple.runs[1].status == BigHO.Failed
    @test ho_tuple.runs[1].post_artefact === nothing # untouched -- this objective was never Stateful

    # Regression: a plain objective returning `missing` is excluded as
    # failed (same treatment as NaN/exceptions), rather than being recorded
    # as a Completed value that later corrupts ranking comparisons.
    let n_calls = Ref(0)
        global missing_after_first(a) = (n_calls[] += 1; n_calls[] == 1 ? 5.0 : missing)
    end
    ho_missing = Hyperoptimizer(missing_after_first, (a=Nominal([1, 2]),); n=2)
    @test_logs (:warn, r"non-Real") run!(ho_missing)
    @test minimum(ho_missing) == 5.0
    @test length(results(ho_missing)) == 1
    @test ho_missing.runs[2].status == BigHO.Failed
end
