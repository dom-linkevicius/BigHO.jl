"""
    DEHBSampler(; F=0.5, crossover=0.5, rng=StableRNG(1))
"""
struct DEHBSampler{T<:Random.AbstractRNG} <: Sampler
    F::Float64
    crossover::Float64
    rng::T
    subpops::Vector{Vector{Int}}
    total_dispatched::Vector{Int}
end
function DEHBSampler(; F::Real=0.5, crossover::Real=0.5, rng::Random.AbstractRNG=StableRNG(1))
    0 < F <= 1 || throw(ArgumentError("F must be in (0, 1], got $F"))
    0 <= crossover <= 1 || throw(ArgumentError("crossover must be in [0,1], got $crossover"))
    return DEHBSampler(Float64(F), Float64(crossover), rng, Vector{Int}[], Int[])
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
    objective isa Stateful && @warn "$(typeof(s)) with a Stateful objective: pre_artefact is only set on promotions in iteration 1; every other trial trains from scratch. post_artefact is recorded either way"

_level(s::DEHB, budget::Int) = _smax(budget, s.r_min, s.η) + 1
_subpop_size(s::DEHB, level::Int) = _capacity(s.R, s.r_min, s.η, level, 1)

function _subpopulations!(s::DEHB)
    de = s.inner
    isempty(de.total_dispatched) || return de
    for level in 1:(_smax(s.R, s.r_min, s.η)+1)
        push!(de.subpops, zeros(Int, _subpop_size(s, level)))
        push!(de.total_dispatched, 0)
    end
    return de
end

_next_slot(s::DEHB, level::Int) = mod(_subpopulations!(s).total_dispatched[level], _subpop_size(s, level)) + 1

_occupants(runs, slots) = Vector{Float64}[runs[id].unit_params for id in slots if id != 0]

_parent_pool(s::DEHB, runs, level::Int) = _occupants(runs, _subpopulations!(s).subpops[level-1])

function _global_pool(s::DEHB, runs)
    pool = Vector{Vector{Float64}}()
    for slots in _subpopulations!(s).subpops
        append!(pool, _occupants(runs, slots))
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
    d.bracket.iteration == 1 ? copy(runs[d.promoted_from].unit_params) :
    _sample_sh_inner(s, candidates, runs, d)

function _entry_for(d::SHDecision{:promote}, s::DEHB, ho, id, params, unit_params)
    d.bracket.iteration == 1 &&
        return @invoke _entry_for(d::SHDecision{:promote}, s::SuccessiveHalving, ho, id, params, unit_params)
    target = d.bracket.rungs[d.rung.rung+1]
    return RunEntry(id, _add_r(params, target.resource), unit_params, _metadata(d.bracket, target.rung))
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:draw})
    level = _level(s, d.rung.resource)
    return _sample_sh_inner(s, candidates, runs, d.bracket, level,
                            _occupants(runs, _subpopulations!(s).subpops[level]))
end

function _sample_sh_inner(s::DEHB, candidates, runs, d::SHDecision{:promote})
    level = _level(s, d.bracket.rungs[d.rung.rung+1].resource)
    return _sample_sh_inner(s, candidates, runs, d.bracket, level, _parent_pool(s, runs, level))
end

function _sample_sh_inner(s::DEHB, candidates, runs, bracket::ActiveBracket, level::Int, parents)
    de = _subpopulations!(s)
    bracket.iteration == 1 && level == 1 && return rand(de.rng, length(candidates))

    occupant = de.subpops[level][_next_slot(s, level)]
    pool = _global_pool(s, runs)

    ### the paper does not seem to mention what happens in the case where there aren't enough targets
    ### whether to take from the same subpopulation, but not replace them or from global population
    target = occupant == 0 ? rand(de.rng, pool) : runs[occupant].unit_params
    append!(parents, rand(de.rng, pool, max(0, 3 - length(parents))))
    return _de_trial(de, target, parents)
end

function create_run_entry(s::DEHB, ho, id, params, unit_params)
    de = _subpopulations!(s)
    entry = @invoke create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    level = _level(s, entry.params.r)
    entry.metadata[:slot] = _next_slot(s, level)
    de.total_dispatched[level] += 1
    return entry
end

function on_tell!(s::DEHB, runs, entry)
    de = _subpopulations!(s)
    if entry.status === Completed
        level = _level(s, entry.params.r)
        slot = entry.metadata[:slot]
        occupant = de.subpops[level][slot]
        if occupant == 0 || entry.value < runs[occupant].value
            de.subpops[level][slot] = entry.id
        end
    end
    return @invoke on_tell!(s::SHSync, runs, entry)
end
