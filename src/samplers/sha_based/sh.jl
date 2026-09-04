# Shared internals for Hyperband/ASHA: successive-halving over one finite pass of brackets, differing only in their promotion-readiness gate (_asha_ready/_hyperband_ready).
# Brackets are independent and never pruned -- every live one is checked for a promotable candidate (oldest first) before a fresh draw starts a new one.

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

# Builds bracket k (1..smax+1); n0/capacities match Li et al. 2018's Algorithm 1 formulas
# (n=⌈(smax+1)η^s/(s+1)⌉, n_i=⌊n·η^-i⌋, s=k-1) bit-for-bit.
function _new_bracket(k::Int, R::Int, r_min::Int, η::Int)
    levels = _resource_levels(R, r_min, η) # levels[i] = r_min*η^(i-1); only ever non-negative exponents
    smax = length(levels) - 1
    resources = levels[(smax-k+2):end] # the top k levels -- k=1 is just [R], k=smax+1 is the whole thing
    n0 = ceil(Int, (smax + 1) * η^(k - 1) / k)
    capacities = [max(1, floor(Int, n0 / η^(i - 1))) for i in 1:k]
    return SHABracket(k, resources, capacities, zeros(Int, k), [Tuple{Int,Float64}[] for _ in 1:k], [Set{Int}() for _ in 1:k])
end

# One-pass total trial count across every bracket -- fully determined by R/η/r_min, no external n.
function _total_trials(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(sum(_new_bracket(k, R, r_min, η).capacities) for k in 1:(smax+1))
end

# Total fresh (non-promoted) draws across the whole run -- `inner`'s own budget, one global
# design covering every bracket's bottom rung.
function _total_draws(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_new_bracket(k, R, r_min, η).capacities[1] for k in 1:(smax+1))
end

# Finds a promotable trial, rungs top-down (top rung excluded, nowhere to promote to);
# `ready(bracket, i)` gates whether rung i is eligible to be checked yet.
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

# n (ho's planned total) is irrelevant here -- inner's budget is _total_draws, from R/η/r_min directly.
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

# Delegates to `inner`, which sees only fresh (non-promoted) entries so far, keeping its own
# row/index bookkeeping (e.g. LHSampler's) aligned with the design it was init'd against.
function _draw_new!(s::SuccessiveHalving, bracket::SHABracket, candidates, runs)
    fresh_runs = filter(e -> get(e.metadata, :rung, nothing) == 1, runs)
    raw_params = s.inner(candidates[2:end], fresh_runs) # skip the reserved :r slot (position 1)
    bracket.dispatched[1] += 1
    s.pending_promoted_id = nothing
    s.pending_rung = 1
    s.pending_bracket_k = bracket.k
    return vcat(bracket.resources[1], raw_params)
end

# Every live bracket is checked for a promotable candidate first (oldest first); only once
# nothing's promotable does a fresh draw happen, advancing to a new bracket if needed.
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

# Not exhausted while anything's promotable in any live bracket (re-checked fresh each call),
# or while a fresh draw/new bracket is still possible.
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
