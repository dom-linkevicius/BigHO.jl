"""
    Sampler

Pluggable candidate-generation strategy. Concrete samplers must implement
the callable interface `(sampler)(ctx::AskContext) -> candidates`, plus
`on_tell!`, `init!`, and `exhausted` -- none of these have a default, so a
missing one is a loud `MethodError` rather than a silent no-op.
"""
abstract type Sampler end

"""
    on_tell!(sampler, entry)

Feedback hook called once per `tell!`, after `entry`'s result is recorded.
"""
function on_tell! end

"""
    init!(sampler, ctx::AskContext)

Called once before the first `ask!`.
"""
function init! end

"""
    exhausted(sampler, ho) -> Bool

Whether the sampler can no longer produce candidates.
"""
function exhausted end
