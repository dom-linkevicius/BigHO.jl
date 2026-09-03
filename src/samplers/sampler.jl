"""
    Sampler

Pluggable candidate-generation strategy. Concrete samplers must implement
the callable interface `(sampler)(ctx::AskContext) -> candidates`, plus
`on_tell!`, `init`, and `exhausted` -- none of these have a default, so a
missing one is a loud `MethodError` rather than a silent no-op.
"""
abstract type Sampler end

"""
    on_tell!(sampler, entry)

Feedback hook called once per `tell!`, after `entry`'s result is recorded.
"""
function on_tell! end

"""
    init(sampler, ctx::AskContext) -> sampler

Called once before the first `ask!`, returning the (possibly new) sampler to actually use -- lets an immutable sampler return a different instance carrying state that depends on `ctx` (e.g. `ho.n`), rather than needing to mutate itself. Not `init!`: it isn't necessarily in-place.
"""
function init end

"""
    exhausted(sampler, ho) -> Bool

Whether the sampler can no longer produce candidates.
"""
function exhausted end
