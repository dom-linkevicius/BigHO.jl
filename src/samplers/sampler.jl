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
    presample_size(sampler) -> Union{Int,Nothing}

`ho.n` if this sampler needs the total sample count fixed up front, else
`nothing`.
"""
presample_size(::Sampler) = nothing

"""
    exhausted(sampler, ho) -> Bool

Whether the sampler can no longer produce candidates.
"""
exhausted(::Sampler, ho) = false
