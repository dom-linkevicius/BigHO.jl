# Shared internals for Hyperband/ASHA -- both are successive-halving over a sequence of
# brackets of decreasing aggressiveness (k = smax+1 rungs, starting at r_min, down to k = 1,
# a single rung at R -- plain random search), run through exactly once (Li et al. 2018's
# Algorithm 1 is a single finite pass, not an indefinite loop) -- so the total trial count
# is fully determined by R/η/r_min alone, no external n needed. They differ only in when a
# promotion is allowed to fire (see _asha_ready in asha.jl / _hyperband_ready in hyperband.jl).
# Brackets are independent: each one's own promotion process runs to completion on its own
# schedule, so every live bracket is checked for a promotable candidate on every ask (oldest
# first), never just the newest one -- only once nothing is promotable anywhere does a fresh
# draw happen, into whichever bracket is currently active. A new bracket becomes active as
# soon as the previous one's bottom rung has been fully DISPATCHED (not necessarily told);
# older brackets are kept around (never pruned) so a still-pending trial's eventual `tell!`,
# and any promotion it later unlocks, still finds its way to the bracket it actually belongs to.

struct SHABracket
    k::Int                                   # number of rungs; k=smax+1 most aggressive, k=1 = random search
    resources::Vector{Int}                   # r_i, bottom (i=1) to top (i=k)
    capacities::Vector{Int}                  # n_i, target dispatch count per rung
    dispatched::Vector{Int}                  # how many drawn so far per rung
    told::Vector{Vector{Tuple{Int,Float64}}} # (trial_id, loss) told so far per rung
    promoted::Vector{Set{Int}}               # trial ids already promoted out of each rung
end

# Hyperband and ASHA share every field; Sync (true/false) only decides which callable/find_promotion gate fires.
mutable struct SuccessiveHalving{Sync} <: Sampler
    R::Int
    r_min::Int
    η::Int
    brackets::Vector{SHABracket}             # creation order; brackets[end] is the one currently accepting fresh draws
    inner::Sampler                           # draws fresh bottom-rung candidates -- any Sampler, e.g. RandomSampler (default) or LHSampler
    pending_promoted_id::Union{Int,Nothing}  # set by the callable, read+cleared by create_run_entry
    pending_rung::Union{Int,Nothing}         # set by the callable, read+cleared by create_run_entry
    pending_bracket_k::Union{Int,Nothing}    # set by the callable, read+cleared by create_run_entry
end
SuccessiveHalving{Sync}(; R::Int, η::Int=3, r_min::Int=1, inner::Sampler=RandomSampler()) where {Sync} =
    SuccessiveHalving{Sync}(R, r_min, η, [_new_bracket(_smax(R, r_min, η) + 1, R, r_min, η)], inner, nothing, nothing, nothing)

_smax(R::Int, r_min::Int, η::Int) = floor(Int, log(R / r_min) / log(η))

_resource_levels(R::Int, r_min::Int, η::Int) = [r_min * η^i for i in 0:_smax(R, r_min, η)]

# Builds bracket k (k=1..smax+1); n0 chosen so this bracket's total resource usage
# (k * n0 * r0) is approximately constant (≈ (smax+1)*R) across brackets, matching the
# original algorithm's design goal. Verified bit-for-bit against Li et al. 2018's own
# Algorithm 1 formulas (n = ⌈(smax+1)·η^s/(s+1)⌉, n_i = ⌊n·η^-i⌋, under s = k-1).
function _new_bracket(k::Int, R::Int, r_min::Int, η::Int)
    levels = _resource_levels(R, r_min, η) # levels[i] = r_min*η^(i-1); only ever non-negative exponents
    smax = length(levels) - 1
    resources = levels[(smax-k+2):end] # the top k levels -- k=1 is just [R], k=smax+1 is the whole thing
    n0 = ceil(Int, (smax + 1) * η^(k - 1) / k)
    capacities = [max(1, floor(Int, n0 / η^(i - 1))) for i in 1:k]
    return SHABracket(k, resources, capacities, zeros(Int, k), [Tuple{Int,Float64}[] for _ in 1:k], [Set{Int}() for _ in 1:k])
end

# The one-pass total trial count across every bracket (k=1..smax+1) -- Hyperband/ASHA's own
# analogue of LHSampler's n: fully determined by R/η/r_min, never given externally.
function _total_trials(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(sum(_new_bracket(k, R, r_min, η).capacities) for k in 1:(smax+1))
end

# Total FRESH draws across the whole run (bottom rung only, every bracket) -- excludes
# promotions, which reuse an existing candidate's params rather than asking `inner` for a new
# one. This is `inner`'s own budget: one global design covering every bracket's bottom rung,
# not a separate one per bracket.
function _total_draws(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_new_bracket(k, R, r_min, η).capacities[1] for k in 1:(smax+1))
end

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

# n (the planned total trial count) isn't relevant here -- inner's own budget is _total_draws
# (fresh bottom-rung draws only, excludes promotions), computed from R/η/r_min directly.
function init(s::SuccessiveHalving, candidates, n)
    s.inner = init(s.inner, candidates[2:end], _total_draws(s.R, s.r_min, s.η)) # drop the reserved :r slot
    return s
end

function _promote!(s::SuccessiveHalving, bracket::SHABracket, runs, rung::Int, promoted_id::Int)
    push!(bracket.promoted[rung], promoted_id)
    new_rung = rung + 1
    params = collect(values(runs[promoted_id].params)[2:end]) # drop the reserved :r slot
    bracket.dispatched[new_rung] += 1
    s.pending_promoted_id = promoted_id
    s.pending_rung = new_rung
    s.pending_bracket_k = bracket.k
    return vcat(bracket.resources[new_rung], params)
end

# Delegates to `inner` for the actual draw -- `inner` sees only fresh (never-promoted) bottom-rung
# entries from every bracket so far, so its own `ask!`-relative bookkeeping (e.g. LHSampler's row
# index) lines up with the single global design it was init'd against.
function _draw_new!(s::SuccessiveHalving, bracket::SHABracket, candidates, runs)
    fresh_runs = filter(e -> get(e.metadata, :rung, nothing) == 1, runs)
    raw_params = s.inner(candidates[2:end], fresh_runs) # skip the reserved :r slot (position 1)
    bracket.dispatched[1] += 1
    s.pending_promoted_id = nothing
    s.pending_rung = 1
    s.pending_bracket_k = bracket.k
    return vcat(bracket.resources[1], raw_params)
end

# Brackets are independent -- each one's own promotion process runs to completion on its own
# schedule, regardless of how many newer brackets have since started. So every live bracket
# (not just the active one) is checked for a promotable candidate first, oldest to newest
# (favoring the most aggressive/longest-running bracket's near-complete configs, matching
# ASHA's own single-bracket preference for finishing over starting). Only once nothing is
# promotable anywhere does a fresh draw happen, into the active bracket -- advancing to a new
# one first if the active bracket's bottom rung is already fully dispatched.
function _ask(s::SuccessiveHalving, ready, candidates, runs)
    for bracket in s.brackets
        promotion = _find_promotion(bracket, s.η, ready)
        promotion !== nothing && return _promote!(s, bracket, runs, promotion...)
    end
    active = s.brackets[end]
    if active.dispatched[1] >= active.capacities[1] && active.k > 1
        active = _new_bracket(active.k - 1, s.R, s.r_min, s.η)
        push!(s.brackets, active)
    end
    return _draw_new!(s, active, candidates, runs)
end

# Not exhausted while anything is currently promotable in any live bracket (checked fresh
# each call, so a later tell! that unblocks an older bracket's promotion is picked back up),
# nor while the active bracket can still take a fresh draw or a new bracket can start.
function _exhausted(s::SuccessiveHalving, ready)
    any(bracket -> _find_promotion(bracket, s.η, ready) !== nothing, s.brackets) && return false
    active = s.brackets[end]
    return active.k == 1 && active.dispatched[1] >= active.capacities[1]
end

function on_tell!(s::SuccessiveHalving, entry)
    entry.status === Completed || return nothing # excluded from ranking via status, never a NaN placeholder
    rung = get(entry.metadata, :rung, nothing)
    bracket_k = get(entry.metadata, :bracket_k, nothing)
    (rung === nothing || bracket_k === nothing) && return nothing
    i = findfirst(b -> b.k == bracket_k, s.brackets)
    i === nothing && return nothing
    push!(s.brackets[i].told[rung], (entry.id, entry.value))
    return nothing
end

function create_run_entry(s::SuccessiveHalving, ho, id, params)
    promoted_id = s.pending_promoted_id
    metadata = Dict{Symbol,Any}(:rung => s.pending_rung, :bracket_k => s.pending_bracket_k)
    s.pending_promoted_id = nothing
    s.pending_rung = nothing
    s.pending_bracket_k = nothing
    promoted_id === nothing && return RunEntry(id, params, metadata)
    return RunEntry(id, params, metadata; pre_artefact=ho.runs[promoted_id].post_artefact)
end
