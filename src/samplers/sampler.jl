abstract type Sampler end

"""
    on_tell!(sampler, entry)

Feedback hook called once per `tell!`, after `entry`'s result is recorded.
"""
on_tell!(::Sampler, entry) = nothing

"""
    init!(sampler, ctx::AskContext)

Called once before the first `ask`.
"""
init!(::Sampler, ctx) = nothing

"""
    exhausted(sampler, ho) -> Bool

Whether the sampler can no longer produce candidates.
"""
exhausted(::Sampler, ho) = false

"""
    FixedPlanSampler

Sampler types whose plan is fixed to `n` at construction (e.g. a Latin
Hypercube design matrix), so `settarget!` can't work for them. Add a type
here when implementing such a sampler.
"""
const FixedPlanSampler = Set{DataType}()
