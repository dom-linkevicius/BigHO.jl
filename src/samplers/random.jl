"""
    RandomSampler(rng=StableRNG(1))

Draw each parameter via `rand(rng, domain)`. Defaults to a `StableRNG`
seeded `1`, reproducible across Julia versions; pass a different `AbstractRNG`
for a different seed or genuine randomness, e.g. `RandomSampler(StableRNG(42))`.
"""
mutable struct RandomSampler{T<:Random.AbstractRNG} <: Sampler
    rng::T
end

RandomSampler() = RandomSampler(StableRNG(1))

function (s::RandomSampler)(ctx::AskContext)
    return [rand(s.rng, d) for d in ctx.candidates]
end
