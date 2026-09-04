"""
    Sampler

Pluggable candidate-generation strategy. Concrete samplers must implement the callable interface `(sampler)(candidates, runs) -> raw_candidates`, plus `on_tell!`, `init`, `exhausted`, `blocked`, and `create_run_entry` -- none have a default, so a missing one is a loud `MethodError`.
`candidates` is `ho.candidates`; `runs` is `ho.runs` (read-only by convention).
"""
abstract type Sampler end

"""
    on_tell!(sampler, runs, entry)

Feedback hook called once per `tell!`, after `entry`'s result is recorded in `runs` (`ho.runs`,
already reflecting this `tell!` -- not read independently, since it may since have changed again).
"""
function on_tell! end

"""
    init(sampler, candidates, n) -> sampler

Called once before the first `ask!`, returning the (possibly new) sampler to actually use -- lets an immutable sampler return a different instance carrying state that depends on `n` (`ho.n`, the planned total trial count, if any), rather than needing to mutate itself. Not `init!`: it isn't necessarily in-place.
"""
function init end

"""
    exhausted(sampler, ho) -> Bool

Whether the sampler can no longer produce candidates. Permanent -- never true, then later false.
"""
function exhausted end

"""
    blocked(sampler, ho) -> Bool

Whether the sampler can't produce a candidate from `ho`'s current state right now, but could
once more trials are told -- unlike `exhausted`, not permanent.
"""
function blocked end

"""
    create_run_entry(sampler, ho, id, params) -> RunEntry

Builds the `RunEntry` for a freshly-asked trial. Most samplers just wrap `params` with no `pre_artefact`; samplers resuming from a prior trial's `post_artefact` seed it accordingly instead.
"""
function create_run_entry end
