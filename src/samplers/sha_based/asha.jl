"""
    ASHA(; R, η=3, r_min=1, inner=RandomSampler())

Asynchronous Hyperband (Li et al. 2020): same one-pass brackets as [`Hyperband`](@ref), but promotes the instant a trial is verifiably top-`1/η` of what's told so far -- never blocks. Multi-bracket scheduling isn't fully specified by the paper, so this is an interpretation, not a transcription.
Construction, `:r`, and the objective-calling convention are identical to [`Hyperband`](@ref).
"""
const ASHA = SuccessiveHalving{false}

# Per Li et al. 2020's get_job(): scan rungs top-down for a trial in the top 1/η told so far, not
# yet promoted, capped by Hyperband's static capacity. Rungs are independent -- no abandonment needed.
function _bracket_decision(s::ASHA, k::Int, runs)
    R, r_min, η = s.R, s.r_min, s.η
    for i in (k-1):-1:1
        told = _told_sorted(runs, k, i)
        n_promotable = min(floor(Int, length(told) / η), _capacity(R, r_min, η, k, i + 1))
        promoted = _promoted_ids(runs, k, i)
        if length(promoted) < n_promotable
            id, _ = first(t for t in told if first(t) ∉ promoted)
            return (:promote, k, i, id)
        end
    end
    _dispatched_count(runs, k, 1) < _capacity(R, r_min, η, k, 1) && return (:draw, k)
    # Bottom rung full and nothing promotable: only conclusive once nothing's still in flight
    # anywhere in the bracket -- otherwise a pending trial could still make something promotable.
    any(i -> _pending_count(runs, k, i) > 0, 1:k) && return (:wait,)
    return k > 1 ? _bracket_decision(s, k - 1, runs) : (:exhausted,)
end

# Whether bracket k can still draw or promote right now -- _bracket_decision's own local
# conditions, without its recursive fall-through to bracket k-1 (a different bracket entirely).
function _bracket_has_room(s::ASHA, k::Int, runs)
    R, r_min, η = s.R, s.r_min, s.η
    _dispatched_count(runs, k, 1) < _capacity(R, r_min, η, k, 1) && return true
    return any(1:(k-1)) do i
        told = _told_sorted(runs, k, i)
        n_promotable = min(floor(Int, length(told) / η), _capacity(R, r_min, η, k, i + 1))
        length(_promoted_ids(runs, k, i)) < n_promotable
    end
end

# Whether rung i is resolved: dispatch finalized (own hard capacity reached, or rung below is
# resolved) and nothing still Pending.
function _rung_resolved(s::ASHA, runs, k::Int, i::Int)
    R, r_min, η = s.R, s.r_min, s.η
    dispatch_final = if _dispatched_count(runs, k, i) >= _capacity(R, r_min, η, k, i)
        true
    elseif i == 1
        false
    else
        _rung_resolved(s, runs, k, i - 1) &&
            _dispatched_count(runs, k, i) >= min(floor(Int, length(_told_sorted(runs, k, i - 1)) / η), _capacity(R, r_min, η, k, i))
    end
    return dispatch_final && _pending_count(runs, k, i) == 0
end

# Warn once when a bracket stalls short of plan -- nothing pending, no rung can accept more.
# One-time transition (pending only decreases), so this fires exactly once.
function on_tell!(s::ASHA, runs, entry)
    k = entry.metadata[:bracket_k]
    R, r_min, η = s.R, s.r_min, s.η

    if all(i -> _pending_count(runs, k, i) == 0, 1:k) && !_bracket_has_room(s, k, runs)
        total_capacity = sum(_capacity(R, r_min, η, k, i) for i in 1:k)
        total_dispatched = sum(_dispatched_count(runs, k, i) for i in 1:k)
        total_dispatched < total_capacity &&
            @warn "$(typeof(s)): bracket $k stalled at $total_dispatched/$total_capacity trials dispatched -- no rung can accept more"
    end

    # Warn once per rung that completes with a failure, checked against the counterfactual with
    # this entry still Pending (a tell can retroactively complete rungs above it too, not just its own).
    before = [e.id == entry.id ? RunEntry(e.id, e.params, e.metadata; pre_artefact=e.pre_artefact) : e for e in runs]
    for i in entry.metadata[:rung]:k
        _rung_resolved(s, runs, k, i) || continue
        _rung_resolved(s, before, k, i) && continue # already known complete before this tell
        _rung_has_failure(runs, k, i) &&
            @warn "$(typeof(s)): rung $i of bracket $k completed with at least one failed trial"
    end
    return nothing
end
