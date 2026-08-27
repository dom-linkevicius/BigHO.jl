abstract type Sampler end

"""
    on_tell!(sampler, entry)

Feedback hook: called once per `tell!`, after `entry::RunEntry`'s result
fields have been recorded, so a sampler can update whatever aggregate state
it maintains internally (e.g. Hyperband's rung bookkeeping, BOHB's KDEs).
Samplers never read `ho.history`/`ho.results`/`ho.runs` directly — this hook
is their only feedback channel.
"""
on_tell!(::Sampler, entry) = nothing

"""
    init!(sampler, ctx::AskContext)

Called once before the first `ask`, so a sampler can validate/precompute
anything that depends on `ctx.candidates`/`ctx.n` (e.g. LHSampler's design matrix).
"""
init!(::Sampler, ctx) = nothing

"""
    presample_size(sampler) -> Union{Int,Nothing}

Return `ho.n` if this sampler requires the total sample count fixed up front
(e.g. Latin Hypercube samplers), or `nothing` if it can `ask` indefinitely.
"""
presample_size(::Sampler) = nothing

"""
    exhausted(sampler, ho) -> Bool

Return `true` once the sampler cannot produce any more candidates
(e.g. a bounded design matrix has been fully claimed).
"""
exhausted(::Sampler, ho) = false
