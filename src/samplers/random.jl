"""
    RandomSampler{T<:AbstractRNG} <: Sampler

Draw each parameter by calling `rand(rng, domain)` on its `Domain` (see
`src/domains.jl`) -- uniformly by default, or according to that `Domain`'s
`weights` if it has any. Optionally pass an `AbstractRNG` to initialize.

`ask` is only ever called from the single driver task in `run!`, so unlike
the pre-rewrite `RandomSampler`, this holds a plain RNG — no `RemoteChannel`
or lock is needed to share it safely across workers, because workers never
call the sampler themselves.

`RandomSampler()` defaults to a `StableRNG` (guaranteed reproducible across
Julia versions, unlike `Random.MersenneTwister`) seeded with a fixed seed of
`1` — reproducible by default, no hidden randomness. If you want a different
seed, pass `RandomSampler(StableRNG(42))`; for genuinely non-reproducible
runs, pass any other `AbstractRNG` explicitly, e.g. `RandomSampler(Random.Xoshiro())`.
"""
mutable struct RandomSampler{T<:Random.AbstractRNG} <: Sampler
    rng::T
end

RandomSampler() = RandomSampler(StableRNG(1))

function (s::RandomSampler)(ctx::AskContext)
    return [rand(s.rng, d) for d in ctx.candidates]
end
