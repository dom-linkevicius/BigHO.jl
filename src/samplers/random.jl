"""
    RandomSampler(rng=StableRNG(1))

Draw each parameter via `rand(rng, domain)`. Default `rng` set to `StableRNG(1)`
"""
struct RandomSampler{T<:Random.AbstractRNG} <: Sampler
    rng::T
end

RandomSampler() = RandomSampler(StableRNG(1))

function (s::RandomSampler)(ctx::AskContext)
    return [rand(s.rng, d) for d in ctx.candidates]
end

on_tell!(::RandomSampler, entry) = nothing
init!(::RandomSampler, ctx) = nothing
exhausted(::RandomSampler, ho) = false
