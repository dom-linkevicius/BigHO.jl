# Shared internals for Hyperband/ASHA -- both are successive-halving over brackets of
# decreasing aggressiveness (k = smax+1 rungs, starting at r_min, down to k = 1, a single
# rung at R -- plain random search). They differ only in when a promotion is allowed to
# fire (see _asha_ready/_hyperband_ready below), so the bracket/rung bookkeeping, the
# top-1/η promotion criterion, and the resource schedule are all written once here.

struct SHABracket
    k::Int                                   # number of rungs; k=smax+1 most aggressive, k=1 = random search
    resources::Vector{Int}                   # r_i, bottom (i=1) to top (i=k)
    capacities::Vector{Int}                  # n_i, target dispatch count per rung (Hyperband only)
    dispatched::Vector{Int}                  # how many drawn so far per rung
    told::Vector{Vector{Tuple{Int,Float64}}} # (trial_id, loss) told so far per rung
    promoted::Vector{Set{Int}}               # trial ids already promoted out of each rung
end

# Hyperband and ASHA share every field; Sync (true/false) only decides which callable/exhausted method fires.
mutable struct SuccessiveHalving{Sync} <: Sampler
    R::Int
    r_min::Int
    η::Int
    bracket::SHABracket
    pending_promoted_id::Union{Int,Nothing} # set by the callable, read+cleared by create_run_entry
    pending_rung::Union{Int,Nothing}        # set by the callable, read+cleared by create_run_entry
end
SuccessiveHalving{Sync}(; R::Int, η::Int=3, r_min::Int=1) where {Sync} =
    SuccessiveHalving{Sync}(R, r_min, η, _new_bracket(_smax(R, r_min, η) + 1, R, r_min, η), nothing, nothing)

_smax(R::Int, r_min::Int, η::Int) = floor(Int, log(R / r_min) / log(η))

_resource_levels(R::Int, r_min::Int, η::Int) = [r_min * η^i for i in 0:_smax(R, r_min, η)]

# Builds bracket k (k=1..smax+1); n0 chosen so this bracket's total resource usage
# (k * n0 * r0) is approximately constant (≈ (smax+1)*R) across brackets, matching the
# original algorithm's design goal (exact per-bracket n_i may differ slightly from Li et
# al. 2018's own ceiling arithmetic, which isn't otherwise specified precisely enough to
# reproduce bit-for-bit).
function _new_bracket(k::Int, R::Int, r_min::Int, η::Int)
    levels = _resource_levels(R, r_min, η) # levels[i] = r_min*η^(i-1); only ever non-negative exponents
    smax = length(levels) - 1
    resources = levels[(smax-k+2):end] # the top k levels -- k=1 is just [R], k=smax+1 is the whole thing
    n0 = ceil(Int, (smax + 1) * η^(k - 1) / k)
    capacities = [max(1, floor(Int, n0 / η^(i - 1))) for i in 1:k]
    return SHABracket(k, resources, capacities, zeros(Int, k), [Tuple{Int,Float64}[] for _ in 1:k], [Set{Int}() for _ in 1:k])
end

_next_bracket_k(current_k::Int, smax::Int) = current_k <= 1 ? smax + 1 : current_k - 1

# Finds a promotable trial, checking rungs top-down (excluding the top rung, which has
# nowhere to promote to) -- ASHA's own get_job() prioritizes advancing near-complete
# configurations over starting new ones at the bottom. `ready(bracket, i)` gates whether
# rung i's current tally is eligible to be checked at all.
function _find_promotion(bracket::SHABracket, η::Int, ready)
    for i in (bracket.k-1):-1:1
        ready(bracket, i) || continue
        told = bracket.told[i]
        kth = length(told) ÷ η
        kth < 1 && continue
        top_ids = Set(first(t) for t in sort(told; by=last)[1:kth])
        for id in top_ids
            id ∉ bracket.promoted[i] && return (i, id)
        end
    end
    return nothing
end

# ASHA: a rung is eligible the moment anything's been told at it (never blocks).
_asha_ready(bracket::SHABracket, i::Int) = !isempty(bracket.told[i])
# Hyperband: only once every trial dispatched at that rung has been told a result.
_hyperband_ready(bracket::SHABracket, i::Int) = length(bracket.told[i]) >= bracket.capacities[i]

"""
    Hyperband(; R, η=3, r_min=1)

The original synchronous Hyperband (Li et al. 2018): loops through brackets of
decreasing aggressiveness (most-to-least), promoting the top `1/η` of a rung to the next
only once every trial dispatched at that rung has been told a result. `smax = ⌊log_η(R/r_min)⌋`
brackets are cycled indefinitely (bounded by `ho.n`, as usual).

Construct via `Hyperoptimizer(objective, candidates, Hyperband(...))`, which prepends a
reserved `:r` candidate (an `Ordinal` over the resource levels `r_min, r_min*η, ..., R`) --
the objective is called as `f(r, params...)` (or, if [`Stateful`](@ref), `f(r, params...; pre_artefact)`,
receiving its own prior `post_artefact` when resumed after a promotion).
"""
const Hyperband = SuccessiveHalving{true}

"""
    ASHA(; R, η=3, r_min=1)

Asynchronous Successive Halving (Li et al. 2020): a single, ever-growing bracket at the
most aggressive setting (`smax+1` rungs starting at `r_min`). Promotes a trial to the next
rung the instant it's verifiably in the top `1/η` of whatever has been told so far at that
rung -- never blocks, unlike [`Hyperband`](@ref).

Construct via `Hyperoptimizer(objective, candidates, ASHA(...))` -- see [`Hyperband`](@ref)
for the reserved `:r` candidate and objective-calling convention, both identical here.
"""
const ASHA = SuccessiveHalving{false}

init(s::SuccessiveHalving, candidates, n) = s

function _promote!(s::SuccessiveHalving, runs, rung::Int, promoted_id::Int)
    bracket = s.bracket
    push!(bracket.promoted[rung], promoted_id)
    new_rung = rung + 1
    params = collect(values(runs[promoted_id].params)[2:end]) # drop the reserved :r slot
    bracket.dispatched[new_rung] += 1
    s.pending_promoted_id = promoted_id
    s.pending_rung = new_rung
    return vcat(bracket.resources[new_rung], params)
end

function _draw_new!(s::SuccessiveHalving, candidates)
    bracket = s.bracket
    raw_params = [rand(d) for d in candidates[2:end]] # skip the reserved :r slot (position 1)
    bracket.dispatched[1] += 1
    s.pending_promoted_id = nothing
    s.pending_rung = 1
    return vcat(bracket.resources[1], raw_params)
end

function (s::Hyperband)(candidates, runs)
    bracket = s.bracket
    promotion = _find_promotion(bracket, s.η, _hyperband_ready)
    if promotion === nothing && all(i -> length(bracket.told[i]) >= bracket.capacities[i], 1:bracket.k)
        s.bracket = _new_bracket(_next_bracket_k(bracket.k, _smax(s.R, s.r_min, s.η)), s.R, s.r_min, s.η)
        bracket = s.bracket
        promotion = _find_promotion(bracket, s.η, _hyperband_ready) # a fresh bracket is never itself resolved
    end
    promotion !== nothing && return _promote!(s, runs, promotion...)
    return _draw_new!(s, candidates)
end

function (s::ASHA)(candidates, runs)
    promotion = _find_promotion(s.bracket, s.η, _asha_ready)
    promotion !== nothing && return _promote!(s, runs, promotion...)
    return _draw_new!(s, candidates)
end

# Bottom-rung dispatch is uncapped for Hyperband too (matches _draw_new!, which never
# checks capacities[1]) -- exhausted() below is what actually paces it via the rung-barrier.
function exhausted(s::Hyperband, ho)
    bracket = s.bracket
    _find_promotion(bracket, s.η, _hyperband_ready) !== nothing && return false
    all(i -> length(bracket.told[i]) >= bracket.capacities[i], 1:bracket.k) && return false # bracket resolved, next ask() advances it
    return bracket.dispatched[1] >= bracket.capacities[1] # true only once the bottom rung stops accepting fresh draws too
end
exhausted(::ASHA, ho) = false

function on_tell!(s::SuccessiveHalving, entry)
    entry.status === Completed || return nothing # excluded from ranking via status, never a NaN placeholder
    rung = get(entry.metadata, :rung, nothing)
    rung === nothing && return nothing
    push!(s.bracket.told[rung], (entry.id, entry.value))
    return nothing
end

function create_run_entry(s::SuccessiveHalving, ho, id, params)
    promoted_id = s.pending_promoted_id
    metadata = Dict{Symbol,Any}(:rung => s.pending_rung)
    s.pending_promoted_id = nothing
    s.pending_rung = nothing
    promoted_id === nothing && return RunEntry(id, params, metadata)
    return RunEntry(id, params, metadata; pre_artefact=ho.runs[promoted_id].post_artefact)
end
