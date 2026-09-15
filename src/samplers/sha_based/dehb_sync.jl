"""
    DEHBSampler(; F=0.5, crossover=0.5, rng=StableRNG(1))
"""
struct DEHBSampler{T<:Random.AbstractRNG} <: Sampler
    F::Float64
    crossover::Float64
    rng::T
end
function DEHBSampler(; F::Real=0.5, crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1))
    0 < F <= 1 || throw(ArgumentError("F must be in (0, 1], got $F"))
    0 <= crossover <= 1 || throw(ArgumentError("crossover must be in [0,1], got $crossover"))
    return DEHBSampler(Float64(F), Float64(crossover), rng)
end

# init is the only hook SuccessiveHalving calls on its inner sampler; everything else is reached
# only by driving a DEHBSampler as a sampler in its own right, which it can't be.
init(s::DEHBSampler, candidates, n) = s

_dehb_standalone() = throw(ArgumentError("DEHBSampler can't be used on its own; construct DEHB(; R, ...) instead"))
on_tell!(::DEHBSampler, runs, entry) = _dehb_standalone()
exhausted(::DEHBSampler, ho) = _dehb_standalone()
blocked(::DEHBSampler, ho) = _dehb_standalone()
create_run_entry(::DEHBSampler, ho, id, params, unit_params) = _dehb_standalone()
(::DEHBSampler)(candidates, runs) = _dehb_standalone()

"""
    DEHB(; R, η=3, r_min=1, iterations=1, F=0.5, crossover=0.5, rng=StableRNG(1))
"""
const DEHB = SuccessiveHalving{true,<:DEHBSampler}

# `DEHB` is a concrete-inner parametrisation, so it doesn't pick up SuccessiveHalving{Sync}'s
# keyword constructor -- give it one that builds the inner sampler from DE's own keywords.
SuccessiveHalving{true,<:DEHBSampler}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1, F::Real=0.5,
                                       crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1)) =
    SuccessiveHalving{true}(; R=R, η=η, r_min=r_min, iterations=iterations,
                            inner=DEHBSampler(; F=F, crossover=crossover, rng=rng))

# §4.1: a budget's subpopulation is the most trials HB ever allocates to it -- that is the first
# rung of the bracket starting there, since every later bracket reaches it only after halvings.
_subpop_size(s::DEHB, budget::Int) = _capacity(s.R, s.r_min, s.η, _smax(budget, s.r_min, s.η) + 1, 1)

function _next_slot(s::DEHB, runs, budget::Int)
    dispatched = count(e -> e.params.r == budget, runs)
    return mod(dispatched, _subpop_size(s, budget)) + 1
end

# Selection only ever replaces a slot with something better, so its survivor is that slot's argmin.
function _subpopulation(s::DEHB, runs, budget::Int)
    slots = Union{Nothing,Vector{Float64}}[nothing for _ in 1:_subpop_size(s, budget)]
    best = fill(Inf, length(slots))
    for e in runs
        e.status === Completed && e.params.r == budget || continue
        j = e.metadata[:slot]
        if e.value < best[j]
            best[j] = e.value
            slots[j] = e.unit_params
        end
    end
    return slots
end

_occupants(slots) = Vector{Float64}[v for v in slots if v !== nothing]

function _global_pool(s::DEHB, runs)
    pool = Vector{Vector{Float64}}()
    for i in 0:_smax(s.R, s.r_min, s.η)
        budget = s.r_min * s.η^i
        append!(pool, _occupants(_subpopulation(s, runs, budget)))
    end
    return pool
end

# DE/rand/1/bin, with `forced` guaranteeing the trial differs from its target somewhere.
function _de_trial(s::DEHBSampler, target::Vector{Float64}, parents::Vector{Vector{Float64}})
    r1, r2, r3 = parents[randperm(s.rng, length(parents))[1:3]]
    mutant = r1 .+ s.F .* (r2 .- r3)
    out_of_range = .!(0 .<= mutant .<= 1)
    mutant[out_of_range] .= rand(s.rng, count(out_of_range))  
    forced = rand(s.rng, eachindex(target))

    cross = rand(s.rng, length(target)) .< s.crossover
    cross[forced] = 1
    return ifelse.(cross, mutant, target)
end

# On the outer sampler: only the schedule knows which budget a fresh draw belongs to.
function _sample_sh_inner(s::DEHB, candidates, runs)
    de = s.inner
    k = _bracket_decision(s, BracketId(1, 1), runs).bracket
    budget = _resource(s.R, s.r_min, s.η, k.index, 1)
    budget == s.r_min && return rand(de.rng, length(candidates))

    slots = _subpopulation(s, runs, budget)
    occupant = slots[_next_slot(s, runs, budget)]
    pool = _global_pool(s, runs)

    ### the paper does not seem to mention what happens in the case where there aren't enough targets
    ### whether to take from the same subpopulation, but not replace them or from global population
    target = occupant === nothing ? rand(de.rng, pool) : occupant 
    parents = _occupants(slots)
    append!(parents, rand(de.rng, pool, max(0, 3 - length(parents))))
    return _de_trial(de, target, parents)
end

function create_run_entry(s::DEHB, ho, id, params, unit_params)
    entry = @invoke create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    entry.metadata[:slot] = _next_slot(s, ho.runs, entry.params.r)
    return entry
end
