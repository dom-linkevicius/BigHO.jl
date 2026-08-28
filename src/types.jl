@enum RunStatus Pending Completed Failed Abandoned

"""
    RunEntry(id, params::NamedTuple, metadata=Dict{Symbol,Any}(); pre_artefact=nothing)

The complete record of one trial: what was asked (`id`, `params`,
`metadata`) and, once told, what happened (`status`, `value`,
`post_artefact`). Immutable -- `tell!` replaces `ho.runs[id]` with a new
`RunEntry` rather than mutating one in place. `value` is `missing` unless
`status === Completed`. `pre_artefact`/`post_artefact` are `nothing` unless
the objective is [`Stateful`](@ref).
"""
struct RunEntry{P<:NamedTuple}
    id::Int
    params::P
    metadata::Dict{Symbol,Any}
    status::RunStatus
    value::Any
    pre_artefact::Any
    post_artefact::Any
end
function RunEntry(id::Int, params::NamedTuple, metadata::Dict{Symbol,Any}=Dict{Symbol,Any}(); pre_artefact=nothing)
    return RunEntry(id, params, metadata, Pending, missing, pre_artefact, nothing)
end

_with_result(entry::RunEntry, status::RunStatus, value, post_artefact) =
    RunEntry(entry.id, entry.params, entry.metadata, status, value, entry.pre_artefact, post_artefact)

"""
    finalize_entry(entry::RunEntry, outcome) -> RunEntry

Classify a raw objective outcome into a new `RunEntry`. Only a non-`NaN`
`Real` becomes `Completed` -- `NaN`, `missing`, a thrown exception, or any
other type all become `Failed` (with a `@warn` showing what was
returned/thrown).
"""
function finalize_entry(entry::RunEntry, outcome::Real)
    if isnan(outcome)
        @warn "Objective returned NaN; excluding this trial as failed" params = entry.params value = outcome
        return _with_result(entry, Failed, missing, entry.post_artefact)
    end
    return _with_result(entry, Completed, outcome, entry.post_artefact)
end
function finalize_entry(entry::RunEntry, outcome)
    @warn "Objective returned a non-Real value; excluding this trial as failed" params = entry.params value = outcome
    return _with_result(entry, Failed, missing, entry.post_artefact)
end

"""
    ObjectiveOutcome(value, post_artefact)

Uniform wrapper [`call_objective`](@ref) returns on normal completion, so
[`finalize_entry`](@ref) can dispatch on this type rather than guess from
the shape of whatever the objective returned.
"""
struct ObjectiveOutcome
    value::Any
    post_artefact::Any
end
function finalize_entry(entry::RunEntry, outcome::ObjectiveOutcome)
    told = finalize_entry(entry, outcome.value) # reuses whichever method matches value's type (Real -> NaN check, or the non-Real fallback)
    return _with_result(told, told.status, told.value, outcome.post_artefact)
end

"""
    Stateful(f)

Wrap an objective that needs `pre_artefact`/`post_artefact` threading (see
[`RunEntry`](@ref)), e.g. `Stateful(train_network)`. `f` is called as
`f(params...; pre_artefact)` and must return `(metric, post_artefact)`.
"""
struct Stateful{F}
    f::F
end

"""
    call_objective(f, params, pre_artefact)

Invoke an objective, returning an [`ObjectiveOutcome`](@ref) on normal
completion. Dispatches on `f`'s type: a plain objective is called as
`f(params...)`; a [`Stateful`](@ref) one threads `pre_artefact` through.
Define your own method on your own wrapper type for custom behavior.
"""
call_objective(f, params, pre_artefact) = ObjectiveOutcome(f(params...), nothing)
call_objective(s::Stateful, params, pre_artefact) = ObjectiveOutcome(s.f(params...; pre_artefact)...)

"""
    AskContext(candidates, n_asked, n)

Read-only context handed to a sampler on `ask`. `n_asked` is trials asked
so far; `n` is the planned total, if any.
"""
struct AskContext
    candidates::Tuple
    n_asked::Int
    n::Union{Int,Nothing}
end
