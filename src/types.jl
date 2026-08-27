@enum RunStatus Pending Completed Failed Abandoned

"""
    RunEntry(id, params::NamedTuple, metadata=Dict{Symbol,Any}(); pre_artefact=nothing)

The complete record of one trial: what was asked (`id`, `params`, `metadata`)
and, once told, what happened (`status`, `value`, `post_artefact`). `RunEntry`
is immutable -- `ask` creates one `Pending`, and `tell!` never mutates it;
instead it replaces `ho.runs[id]` with a new `RunEntry` computed by
[`apply_outcome`](@ref). Nothing else ever writes to `ho.runs`, so that
single replacement is still the sole place a trial's recorded outcome
changes -- immutability just means it happens by replacement, not by writing
to fields in place.

`metadata` is opaque, sampler-owned scheduling context (e.g. Hyperband's
budget/bracket/rung) -- the core never interprets it. (Being a `Dict`, its
*contents* remain mutable in place regardless of `RunEntry` itself being
immutable, if a sampler needs that.)

`pre_artefact`/`post_artefact` are for objectives that carry state a plain
`(params...) -> value` return can't express (e.g. a trained network with
optimizer/schedule state that can't be cheaply reconstructed from `params`
alone) -- see [`Stateful`](@ref). `pre_artefact` is set at `ask` time (by a
sampler that wants to resume/continue from a prior trial's `post_artefact`;
always `nothing` unless a sampler explicitly sets it) and `post_artefact` at
`tell!` time, from whatever the objective returned alongside its value. Both
are simply `nothing` whenever there's no stateful object at all -- the
common case, requiring no special handling anywhere.

`value` is `missing` whenever `status !== Completed`. A trial's `status` is
the sole signal for success/failure. `NaN` is excluded as a valid
optimization value -- an objective returning `NaN` produces a `Failed`
entry, the same as one that throws (see [`apply_outcome`](@ref)).
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
    apply_outcome(entry::RunEntry, outcome) -> RunEntry

Classify a raw objective outcome and return a new `RunEntry` recording it
(`entry` itself is untouched -- `RunEntry` is immutable, see its docstring).
`NaN` and `missing` values and thrown exceptions are all treated as failures
-- excluded as valid optimization values, not silently recorded -- and each
emits a `@warn` showing `entry.params` alongside what was returned/thrown,
so a failing trial is never silent and is easy to reproduce.

`outcome::ObjectiveOutcome` (what `call_objective` always produces on normal
completion) is unpacked by recursing into whichever method matches its
`.value` -- this is what makes `Stateful`'s artefact-unpacking unambiguous:
dispatch is on this dedicated wrapper *type*, never inferred from the
*shape* of whatever the user's objective actually returned, so a plain
objective that happens to legitimately return its own 2-tuple is never
confused with `(metric, post_artefact)`.

Currently unhandled: `Abandoned`, since no executor produces it yet (that's
fault-tolerant-executor future work) -- if one ever does, this needs a
matching method here, or it would silently fall through to the generic case
below and be misclassified as `Completed`.
"""
apply_outcome(entry::RunEntry, outcome) = _with_result(entry, Completed, outcome, entry.post_artefact)
function apply_outcome(entry::RunEntry, outcome::Real)
    if isnan(outcome)
        @warn "Objective returned NaN; excluding this trial as failed" params = entry.params value = outcome
        return _with_result(entry, Failed, missing, entry.post_artefact)
    end
    return _with_result(entry, Completed, outcome, entry.post_artefact)
end
function apply_outcome(entry::RunEntry, outcome::Missing)
    @warn "Objective returned missing; excluding this trial as failed" params = entry.params
    return _with_result(entry, Failed, missing, entry.post_artefact)
end
function apply_outcome(entry::RunEntry, outcome::Exception)
    @warn "Objective threw an exception; excluding this trial as failed" params = entry.params exception = outcome
    return _with_result(entry, Failed, missing, entry.post_artefact)
end

"""
    ObjectiveOutcome(value, post_artefact)

Internal, uniform wrapper [`call_objective`](@ref) always returns on normal
completion (never on a thrown exception, which bypasses this entirely).
Existing purely so [`apply_outcome`](@ref) can dispatch on a dedicated
*type* rather than structurally guessing "is this a `(metric, post_artefact)`
pair" from an arbitrary returned tuple's shape.
"""
struct ObjectiveOutcome
    value::Any
    post_artefact::Any
end
function apply_outcome(entry::RunEntry, outcome::ObjectiveOutcome)
    told = apply_outcome(entry, outcome.value) # reuses whichever method matches value's type (Real -> NaN check, Missing, or the generic case)
    return _with_result(told, told.status, told.value, outcome.post_artefact)
end

"""
    Stateful(f)

Wrap an objective that needs `pre_artefact`/`post_artefact` threading (see
[`RunEntry`](@ref)) -- e.g. `Stateful(train_network)`. The wrapped function
is called as `f(params...; pre_artefact)` (always -- `pre_artefact` is
`nothing` unless a sampler set one) and must return `(metric, post_artefact)`
rather than a bare value.

A plain, unwrapped objective (the default, and the common case when there's
no stateful object at all) is unaffected: it's still called as
`f(params...)`, exactly as before `Stateful` existed.
"""
struct Stateful{F}
    f::F
end

"""
    call_objective(f, params, pre_artefact)

Invoke an objective, returning an [`ObjectiveOutcome`](@ref) on normal
completion (a thrown exception propagates instead, to be caught by
[`safe_call`](@ref)). Dispatches on `f`'s type so a plain objective (the
default) is called exactly as `f(params...)`, ignoring `pre_artefact`
entirely -- `Stateful`-wrapped objectives thread it through instead. Define
your own method on your own wrapper type for custom artefact-handling
behavior beyond what `Stateful`'s opaque passthrough provides (your method
doesn't have to return `ObjectiveOutcome` at all if you don't need artefact
threading -- `apply_outcome` handles plain values/exceptions regardless).
"""
call_objective(f, params, pre_artefact) = ObjectiveOutcome(f(params...), nothing)
call_objective(s::Stateful, params, pre_artefact) = ObjectiveOutcome(s.f(params...; pre_artefact)...)

"""
    AskContext(candidates, n_asked, n)

Read-only context handed to a sampler on `ask` — samplers never see `ho` itself.
`n_asked` is the number of trials asked so far (0-based), `n` is the planned
total sample count if the `Hyperoptimizer` was given one.
"""
struct AskContext
    candidates::Tuple
    n_asked::Int
    n::Union{Int,Nothing}
end
