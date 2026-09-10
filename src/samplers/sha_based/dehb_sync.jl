"""
    DEHBSampler(; F=0.5, crossover=0.9, rng=StableRNG(1))

Differential-evolution inner sampler: draws each fresh bottom-rung candidate by evolving the
configurations already told at that rung's budget, rather than independently at random.
`F` scales the difference vector, `crossover` is the binomial crossover rate. Used via
[`DEHB`](@ref); falls back to a random draw until the budget has enough completed trials to
evolve from.
"""
struct DEHBSampler{T<:Random.AbstractRNG} <: Sampler
    F::Float64
    crossover::Float64
    rng::T
end
function DEHBSampler(; F::Real=0.5, crossover::Real=0.9, rng::Random.AbstractRNG=StableRNG(1))
    F > 0 || throw(ArgumentError("F must be positive, got $F"))
    0 <= crossover <= 1 || throw(ArgumentError("crossover must be in [0,1], got $crossover"))
    return DEHBSampler(Float64(F), Float64(crossover), rng)
end

on_tell!(::DEHBSampler, runs, entry) = nothing
init(s::DEHBSampler, candidates, n) = s
exhausted(::DEHBSampler, ho) = false
blocked(::DEHBSampler, ho) = false
create_run_entry(::DEHBSampler, ho, id, params) = RunEntry(id, params)

# Only ever reached if used outside a SuccessiveHalving, where there are no rung budgets to
# evolve within -- the useful path is _sample_sh_inner(s::DEHB, ...) below.
(s::DEHBSampler)(candidates, runs) = [rand(s.rng, d) for d in candidates]

"""
    DEHB(; R, η=3, r_min=1, F=0.5, crossover=0.9, rng=StableRNG(1))

DEHB (Awad et al. 2021): [`Hyperband`](@ref)'s synchronous schedule with a [`DEHBSampler`](@ref)
inner, so fresh candidates are evolved from what the same budget already learned instead of drawn
independently -- adaptive, with no density model to fit (unlike BOHB's TPE).
Promotion is unchanged from `Hyperband`, so [`Stateful`](@ref) still applies to it as usual.
"""
const DEHB = SuccessiveHalving{true,<:DEHBSampler}

# `DEHB` is a concrete-inner parametrisation, so it doesn't pick up SuccessiveHalving{Sync}'s
# keyword constructor -- give it one that builds the inner sampler from DE's own keywords.
SuccessiveHalving{true,<:DEHBSampler}(; R::Int, η::Int=3, r_min::Int=1, F::Real=0.5, crossover::Real=0.9,
                                       rng::Random.AbstractRNG=StableRNG(1)) =
    SuccessiveHalving{true}(; R=R, η=η, r_min=r_min, inner=DEHBSampler(; F=F, crossover=crossover, rng=rng))

# Unit-hypercube encoding by candidate index -- the bin centre of a value's position in the
# domain's own candidate list, so DE arithmetic respects whatever spacing the user declared
# (uniform in [0,1] over a log grid stays uniform in log space).
# PROVISIONAL: to be replaced by Domain's own encoding once that lands.
_to_unit(d::Domain, v) = (findfirst(==(v), d.values) - 0.5) / length(d.values)
_from_unit(d::Domain, u::Real) = d.values[clamp(ceil(Int, u * length(d.values)), 1, length(d.values))]
_to_unit(dims::Tuple, params::NamedTuple) = Float64[_to_unit(d, v) for (d, v) in zip(dims, values(params))]
_from_unit(dims::Tuple, u::AbstractVector{<:Real}) = [_from_unit(d, ui) for (d, ui) in zip(dims, u)]

# DE/rand/1/bin: mutant = x_r1 + F(x_r2 - x_r3) over three distinct members, then binomial
# crossover with the target, with one coordinate always taken from the mutant so a trial can
# never come out identical to its target.
function _de_trial(s::DEHBSampler, pool::Vector{Vector{Float64}})
    idx = randperm(s.rng, length(pool))
    target, r1, r2, r3 = pool[idx[1]], pool[idx[2]], pool[idx[3]], pool[idx[4]]
    mutant = clamp.(r1 .+ s.F .* (r2 .- r3), 0.0, 1.0)
    forced = rand(s.rng, eachindex(target))
    return [(j == forced || rand(s.rng) < s.crossover) ? mutant[j] : target[j] for j in eachindex(target)]
end

# Dispatched on the outer sampler because the population to evolve is budget-specific, and only
# the outer sampler knows the schedule: a fresh draw for bracket k sits at rung 1's resource
# level, so that level's completed trials are the subpopulation DEHB evolves.
function _sample_sh_inner(s::DEHB, candidates, runs)
    de, dims = s.inner, _drop_r(candidates)
    k = _bracket_decision(s, _smax(s.R, s.r_min, s.η) + 1, runs)[2]
    budget = _resource(s.R, s.r_min, s.η, k, 1)
    pool = [_to_unit(dims, _drop_r(e.params)) for e in runs if e.status === Completed && e.params.r == budget]
    # DE/rand/1 needs a target plus three distinct parents; below that, initialise at random.
    length(pool) >= 4 || return [rand(de.rng, d) for d in dims]
    return _from_unit(dims, _de_trial(de, pool))
end
