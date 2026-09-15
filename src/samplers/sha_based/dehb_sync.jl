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

SuccessiveHalving{true,<:DEHBSampler}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1, F::Real=0.5,
                                       crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1)) =
    SuccessiveHalving{true}(; R=R, η=η, r_min=r_min, iterations=iterations,
                            inner=DEHBSampler(; F=F, crossover=crossover, rng=rng))

_check_objective(s::DEHB, objective) =
    objective isa Stateful && @warn "$(typeof(s)) with a Stateful objective: pre_artefact is only set on promotions in brackets above bracket 1 of iteration 1; every other trial trains from scratch. post_artefact is recorded either way"

_subpop_size(s::DEHB, budget::Int) = _capacity(s.R, s.r_min, s.η, _smax(budget, s.r_min, s.η) + 1, 1)

function _next_slot(s::DEHB, runs, budget::Int)
    dispatched = count(e -> e.params.r == budget, runs)
    return mod(dispatched, _subpop_size(s, budget)) + 1
end

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

_parent_pool(s::DEHB, runs, budget::Int) = _occupants(_subpopulation(s, runs, budget ÷ s.η))

function _global_pool(s::DEHB, runs)
    pool = Vector{Vector{Float64}}()
    for i in 0:_smax(s.R, s.r_min, s.η)
        budget = s.r_min * s.η^i
        append!(pool, _occupants(_subpopulation(s, runs, budget)))
    end
    return pool
end

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

_propose(d::SHDecision{:promote}, s::DEHB, candidates, runs) =
    d.bracket_id.iteration == 1 && d.bracket_id.bracket > 1 ? copy(runs[d.promoted_from].unit_params) :
    _sample_sh_inner(s, candidates, runs, d)

function _entry_for(d::SHDecision{:promote}, s::DEHB, ho, id, params, unit_params)
    d.bracket_id.iteration == 1 && d.bracket_id.bracket > 1 &&
        return @invoke _entry_for(d::SHDecision{:promote}, s::SuccessiveHalving, ho, id, params, unit_params)
    with_r = _add_r(params, _resource(s.R, s.r_min, s.η, d.bracket_id.bracket, d.rung + 1))
    return RunEntry(id, with_r, unit_params, Dict{Symbol,Any}(:rung => d.rung + 1, :bracket => d.bracket_id))
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:draw})
    budget = _resource(s.R, s.r_min, s.η, d.bracket_id.bracket, d.rung)
    slots = _subpopulation(s, runs, budget)
    return _sample_sh_inner(s, candidates, runs, d.bracket_id, budget, slots, _occupants(slots))
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:promote})
    budget = _resource(s.R, s.r_min, s.η, d.bracket_id.bracket, d.rung + 1)
    slots = _subpopulation(s, runs, budget)
    return _sample_sh_inner(s, candidates, runs, d.bracket_id, budget, slots, _parent_pool(s, runs, budget))
end

function _sample_sh_inner(s::DEHB, candidates, runs, k::BracketId, budget::Int, slots, parents)
    de = s.inner
    k.iteration == 1 && budget == s.r_min && return rand(de.rng, length(candidates))

    occupant = slots[_next_slot(s, runs, budget)]
    pool = _global_pool(s, runs)

    ### the paper does not seem to mention what happens in the case where there aren't enough targets
    ### whether to take from the same subpopulation, but not replace them or from global population
    target = occupant === nothing ? rand(de.rng, pool) : occupant
    append!(parents, rand(de.rng, pool, max(0, 3 - length(parents))))
    return _de_trial(de, target, parents)
end

function create_run_entry(s::DEHB, ho, id, params, unit_params)
    entry = @invoke create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    entry.metadata[:slot] = _next_slot(s, ho.runs, entry.params.r)
    return entry
end
