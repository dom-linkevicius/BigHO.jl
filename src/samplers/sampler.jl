"""
    Sampler

Pluggable candidate-generation strategy. Concrete samplers must implement
the callable interface `(sampler)(candidates, runs) -> raw_candidates`, plus
`on_tell!`, `init`, `exhausted`, and `create_run_entry` -- none of these have
a default, so a missing one is a loud `MethodError` rather than a silent no-op.
`candidates` is `ho.candidates` (the `Domain`s, in order); `runs` is `ho.runs`
(read-only by convention -- trials asked so far, including still-`Pending` ones).
"""
abstract type Sampler end

"""
    on_tell!(sampler, entry)

Feedback hook called once per `tell!`, after `entry`'s result is recorded.
"""
function on_tell! end

"""
    init(sampler, candidates, n) -> sampler

Called once before the first `ask!`, returning the (possibly new) sampler to actually use -- lets an immutable sampler return a different instance carrying state that depends on `n` (`ho.n`, the planned total trial count, if any), rather than needing to mutate itself. Not `init!`: it isn't necessarily in-place.
"""
function init end

"""
    exhausted(sampler, ho) -> Bool

Whether the sampler can no longer produce candidates.
"""
function exhausted end

"""
    create_run_entry(sampler, ho, id, params) -> RunEntry

Builds the `RunEntry` for a freshly-asked trial. Most samplers just wrap `params` with no
`pre_artefact`; samplers whose trials need to resume from a prior trial's `post_artefact`
seed it accordingly instead.
"""
function create_run_entry end
