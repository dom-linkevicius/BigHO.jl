# Hyperband and ASHA share this exact shape; Sync (true/false) only decides which _bracket_decision
# applies -- ASHA's own variant isn't implemented yet. Immutable: bracket/rung progress is derived
# fresh from `runs` each time, nothing cached, so there's nothing else to store beyond the fixed hyperparameters.
struct SuccessiveHalving{Sync} <: Sampler
    R::Int
    r_min::Int
    η::Int
    inner::Sampler # draws fresh bottom-rung candidates -- any Sampler, e.g. RandomSampler (default) or LHSampler
end
function SuccessiveHalving{Sync}(; R::Int, η::Int=3, r_min::Int=1, inner::Sampler=RandomSampler()) where {Sync}
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    η > 1 || throw(ArgumentError("η must be greater than 1, got $η"))
    r_min > 0 || throw(ArgumentError("r_min must be positive, got $r_min"))
    r_min <= R || throw(ArgumentError("r_min must be <= R, got r_min=$r_min, R=$R"))
    return SuccessiveHalving{Sync}(R, r_min, η, inner)
end
# Shared pure math for Hyperband/ASHA's bracket/rung schedule -- fully determined by R/η/r_min alone.

# ndigits(n; base) computes ⌊log_base(n)⌋+1 exactly (integer arithmetic, no floating-point log),
# unlike floor(log(R/r_min)/log(η)), which misrounds on exact powers of η (e.g. R=243, η=3).
# R÷r_min (not R/r_min) is safe here: floor is monotonic, so no integer power of η can fall
# strictly between ⌊R/r_min⌋ and the true ratio R/r_min.
_smax(R::Int, r_min::Int, η::Int) = ndigits(R ÷ r_min; base=η) - 1

_resource_levels(R::Int, r_min::Int, η::Int) = [r_min * η^i for i in 0:_smax(R, r_min, η)]

# n_i target dispatch count for rung i (bottom=1..top=k) of bracket k (1..smax+1). Verified
# bit-for-bit against Li et al. 2018's Algorithm 1 (n=⌈(smax+1)η^s/(s+1)⌉, n_i=⌊n·η^-i⌋, s=k-1).
function _capacity(R::Int, r_min::Int, η::Int, k::Int, i::Int)
    smax = _smax(R, r_min, η)
    n0 = ceil(Int, (smax + 1) * η^(k - 1) / k)
    return max(1, floor(Int, n0 / η^(i - 1)))
end

# r_i resource level for rung i of bracket k.
_resource(R::Int, r_min::Int, η::Int, k::Int, i::Int) = r_min * η^(_smax(R, r_min, η) - k + i)

# candidates/params always carry a reserved :r name -- looked up by key rather than assumed to be
# at a fixed position, so removing it never depends on where adding it put it.
_add_r(candidates::NamedTuple, r_domain) = merge(NamedTuple{(:r,)}((r_domain,)), candidates)
_drop_r(params::NamedTuple) = params[filter(!=(:r), keys(params))]

# How many trials exist (any status) at rung i of bracket k -- an ask! happened, regardless of outcome.
_dispatched_count(runs, k::Int, i::Int) =
    count(e -> get(e.metadata, :bracket_k, nothing) == k && get(e.metadata, :rung, nothing) == i, runs)

# Completed trials at rung i of bracket k, sorted best (lowest loss) first.
_told_sorted(runs, k::Int, i::Int) =
    sort([(e.id, e.value) for e in runs if e.status === Completed && get(e.metadata, :bracket_k, nothing) == k && get(e.metadata, :rung, nothing) == i]; by=last)

# Still in-flight (dispatched, not yet told) trials at rung i of bracket k.
_pending_count(runs, k::Int, i::Int) =
    count(e -> e.status === Pending && get(e.metadata, :bracket_k, nothing) == k && get(e.metadata, :rung, nothing) == i, runs)

# Rung i's effective dispatch target: rung 1 is always the fixed capacity; for i>1, it's capped
# by however many actually survived rung i-1 (which may be short of capacity due to failures).
# nothing if rung i-1 hasn't resolved yet, so the target isn't knowable.
function _rung_target(R::Int, r_min::Int, η::Int, runs, k::Int, i::Int)
    i == 1 && return _capacity(R, r_min, η, k, 1)
    _rung_resolved(R, r_min, η, runs, k, i - 1) || return nothing
    return min(_capacity(R, r_min, η, k, i), length(_told_sorted(runs, k, i - 1)))
end

# Whether rung i of bracket k is done: fully dispatched to its (possibly shrunk) target, nothing
# still Pending -- i.e. its final told count is now known and will never change again.
function _rung_resolved(R::Int, r_min::Int, η::Int, runs, k::Int, i::Int)
    target = _rung_target(R, r_min, η, runs, k, i)
    target === nothing && return false
    return _dispatched_count(runs, k, i) >= target && _pending_count(runs, k, i) == 0
end

# One-pass total trial count across every bracket -- fully determined by R/η/r_min, no external n.
function _total_trials(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_capacity(R, r_min, η, k, i) for k in 1:(smax+1) for i in 1:k)
end

# Total fresh (non-promoted) draws across the whole run -- `inner`'s own budget.
function _total_draws(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_capacity(R, r_min, η, k, 1) for k in 1:(smax+1))
end

# Shared across every SuccessiveHalving sampler -- the only sampler-specific piece is
# _sample_sh_inner (dispatched per type, e.g. hyperband.jl's), which does the actual sampling
# for a fresh draw (Hyperband/ASHA delegate to `inner`; BOHB, later, won't).
function (s::SuccessiveHalving)(candidates, runs)
    action = _bracket_decision(s, _smax(s.R, s.r_min, s.η) + 1, runs)
    if action[1] === :draw
        k = action[2]
        raw_params = _sample_sh_inner(s, candidates, runs)
        return vcat(_resource(s.R, s.r_min, s.η, k, 1), raw_params)
    end
    k, i, promoted_id = action[2], action[3], action[4]
    params = collect(_drop_r(runs[promoted_id].params)) # drop the reserved :r slot, by key not position
    return vcat(_resource(s.R, s.r_min, s.η, k, i + 1), params)
end

# How a fresh draw is handed to `inner` -- most samplers don't care what `runs` they're given.
_sample_sh_inner(inner::Sampler, candidates, runs) = inner(candidates[2:end], runs) # skip the reserved :r slot

# LHSampler is row-indexed off length(runs) -- needs only fresh (never-promoted) draws so its
# row index stays aligned with the design it was built for, not inflated by promotions.
_sample_sh_inner(inner::LHSampler, candidates, runs) =
    inner(candidates[2:end], filter(e -> get(e.metadata, :rung, nothing) == 1, runs)) # skip the reserved :r slot

init(s::SuccessiveHalving, candidates, n) =
    typeof(s)(s.R, s.r_min, s.η, init(s.inner, candidates[2:end], _total_draws(s.R, s.r_min, s.η)))

# _bracket_decision is pure (never calls `inner`), so exhausted/blocked/create_run_entry can
# safely re-derive the same decision the callable already resolved for this ask!, without
# re-triggering a stateful sampler like RandomSampler a second time.
exhausted(s::SuccessiveHalving, ho) =
    first(_bracket_decision(s, _smax(s.R, s.r_min, s.η) + 1, ho.runs)) === :exhausted

# Blocked, not exhausted: the current rung is at dispatch capacity but not yet fully told --
# more candidates exist, just not until an outstanding trial is told.
blocked(s::SuccessiveHalving, ho) =
    first(_bracket_decision(s, _smax(s.R, s.r_min, s.η) + 1, ho.runs)) === :wait

function create_run_entry(s::SuccessiveHalving, ho, id, params)
    action = _bracket_decision(s, _smax(s.R, s.r_min, s.η) + 1, ho.runs)
    if action[1] === :draw
        k = action[2]
        return RunEntry(id, params, Dict{Symbol,Any}(:rung => 1, :bracket_k => k))
    end
    k, i, promoted_id = action[2], action[3], action[4]
    metadata = Dict{Symbol,Any}(:rung => i + 1, :bracket_k => k)
    return RunEntry(id, params, metadata; pre_artefact=ho.runs[promoted_id].post_artefact)
end
