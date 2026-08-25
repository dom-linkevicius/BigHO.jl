"""
    RandomSampler{T<:AbstractRNG} <: Sampler

Sample a value for each parameter uniformly at random from the candidate
vectors. Log-uniform sampling is available by providing a log-spaced
candidate vector. Optionally pass an `AbstractRNG` to initialize.

`ask` is only ever called from the single driver task in `run!`, so unlike
the pre-rewrite `RandomSampler`, this holds a plain RNG — no `RemoteChannel`
or lock is needed to share it safely across workers, because workers never
call the sampler themselves.
"""
mutable struct RandomSampler{T<:Random.AbstractRNG} <: Sampler
    rng::T
end

RandomSampler() = RandomSampler(Random.MersenneTwister(rand(1:1000)))

function (s::RandomSampler)(ctx::AskContext)
    return [list[rand(s.rng, 1:length(list))] for list in ctx.candidates]
end
