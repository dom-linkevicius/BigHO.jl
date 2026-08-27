"""
    Trial(id, params, metadata=Dict{Symbol,Any}())

An immutable record of one sampled candidate. `metadata` is opaque, sampler-owned
scheduling context (e.g. Hyperband's budget/bracket/rung and resumable `state`) —
the core never interprets it.
"""
struct Trial{P<:Tuple}
    id::Int
    params::P
    metadata::Dict{Symbol,Any}
end
Trial(id::Int, params::Tuple) = Trial(id, params, Dict{Symbol,Any}())

@enum TrialStatus Completed Failed Abandoned

"""
    Result(value, status)

`value` is `missing` whenever `status !== Completed`. A trial's `status` is the
sole signal for success/failure — `NaN` is never used as a failure sentinel,
since an objective may legitimately return `NaN` as a real value.
"""
struct Result
    value::Any
    status::TrialStatus
end

to_result(outcome) = Result(outcome, Completed)
to_result(outcome::Exception) = Result(missing, Failed)
to_result(outcome::Result) = outcome # already a Result (e.g. from a fault-tolerant executor) -- pass through unchanged

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
