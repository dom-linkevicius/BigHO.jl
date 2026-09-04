"""
    RandomSampler(rng=StableRNG(1))

Draw each parameter via `rand(rng, domain)`. Default `rng` set to `StableRNG(1)`
"""
struct RandomSampler{T<:Random.AbstractRNG} <: Sampler
    rng::T
end

RandomSampler() = RandomSampler(StableRNG(1))

function (s::RandomSampler)(candidates, runs)
    return [rand(s.rng, d) for d in candidates]
end

on_tell!(::RandomSampler, runs, entry) = nothing
init(s::RandomSampler, candidates, n) = s
exhausted(::RandomSampler, ho) = false
blocked(::RandomSampler, ho) = false
create_run_entry(::RandomSampler, ho, id, params) = RunEntry(id, params)
