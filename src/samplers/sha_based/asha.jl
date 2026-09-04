"""
    ASHA(; R, η=3, r_min=1, inner=RandomSampler())

Asynchronous Hyperband (Li et al. 2020): same one-pass brackets as [`Hyperband`](@ref), but promotes the instant a trial is verifiably top-`1/η` of what's told so far -- never blocks. Multi-bracket scheduling isn't fully specified by the paper, so this is an interpretation, not a transcription.
Construction, `:r`, and the objective-calling convention are identical to [`Hyperband`](@ref).
"""
const ASHA = SuccessiveHalving{false}

# Per Li et al. 2020 Algorithm 2's get_job(): scan rungs top-down (excluding the top rung --
# nothing promotes out of it) for a promotable trial, i.e. one verifiably in the top 1/η of
# what's told so far at that rung and not already promoted; promote the first one found. Capped
# by the same static per-rung capacity Hyperband uses (⌊told/η⌋ alone is unbounded), so ho.n
# stays exact. Only once nothing anywhere is promotable does it fall back to a fresh draw.
# Rungs are otherwise independent -- unlike Hyperband, a rung with zero survivors just never
# promotes past it; it doesn't block or need to abandon anything below it.
function _bracket_decision(s::ASHA, k::Int, runs)
    R, r_min, η = s.R, s.r_min, s.η
    for i in (k-1):-1:1
        told = _told_sorted(runs, k, i)
        n_promotable = min(floor(Int, length(told) / η), _capacity(R, r_min, η, k, i + 1))
        n_promoted = _dispatched_count(runs, k, i + 1)
        n_promoted < n_promotable && return (:promote, k, i, first(told[n_promoted+1]))
    end
    _dispatched_count(runs, k, 1) < _capacity(R, r_min, η, k, 1) && return (:draw, k)
    # Bottom rung full and nothing promotable: only conclusive once nothing's still in flight
    # anywhere in the bracket -- otherwise a pending trial could still make something promotable.
    any(i -> _pending_count(runs, k, i) > 0, 1:k) && return (:wait,)
    return k > 1 ? _bracket_decision(s, k - 1, runs) : (:exhausted,)
end

# ASHA's own sampling strategy for a fresh draw: delegate to `inner`, same as Hyperband's.
_sample_sh_inner(s::ASHA, candidates, runs) = _sample_sh_inner(s.inner, candidates, runs)

# Whether bracket k can still draw or promote right now -- _bracket_decision's own local
# conditions, without its recursive fall-through to bracket k-1 (a different bracket entirely).
function _bracket_has_room(s::ASHA, k::Int, runs)
    R, r_min, η = s.R, s.r_min, s.η
    _dispatched_count(runs, k, 1) < _capacity(R, r_min, η, k, 1) && return true
    return any(1:(k-1)) do i
        told = _told_sorted(runs, k, i)
        n_promotable = min(floor(Int, length(told) / η), _capacity(R, r_min, η, k, i + 1))
        _dispatched_count(runs, k, i + 1) < n_promotable
    end
end

# Whether rung i's dispatch count is finalized (won't grow anymore): either it's already at its
# own hard capacity, or the rung below it is itself resolved (so its told count -- and thus how
# many survivors it can feed forward -- is now fixed too).
function _rung_dispatch_final(s::ASHA, runs, k::Int, i::Int)
    R, r_min, η = s.R, s.r_min, s.η
    _dispatched_count(runs, k, i) >= _capacity(R, r_min, η, k, i) && return true
    i == 1 && return false
    _rung_resolved(s, runs, k, i - 1) || return false
    n_promotable = min(floor(Int, length(_told_sorted(runs, k, i - 1)) / η), _capacity(R, r_min, η, k, i))
    return _dispatched_count(runs, k, i) >= n_promotable
end

_rung_resolved(s::ASHA, runs, k::Int, i::Int) = _rung_dispatch_final(s, runs, k, i) && _pending_count(runs, k, i) == 0

_rung_has_failure(runs, k::Int, i::Int) =
    any(e -> e.status === Failed && get(e.metadata, :bracket_k, nothing) == k && get(e.metadata, :rung, nothing) == i, runs)

# Warn only when a whole bracket stalls short of its full plan -- nothing pending anywhere in
# it (so nothing can still change) and no rung can accept another draw or promotion. This is a
# one-time, bracket-wide transition (pending counts only ever decrease), so it fires exactly once.
function on_tell!(s::ASHA, runs, entry)
    k = entry.metadata[:bracket_k]
    R, r_min, η = s.R, s.r_min, s.η

    if all(i -> _pending_count(runs, k, i) == 0, 1:k) && !_bracket_has_room(s, k, runs)
        total_capacity = sum(_capacity(R, r_min, η, k, i) for i in 1:k)
        total_dispatched = sum(_dispatched_count(runs, k, i) for i in 1:k)
        total_dispatched < total_capacity &&
            @warn "$(typeof(s)): bracket $k stalled at $total_dispatched/$total_capacity trials dispatched -- no rung can accept more"
    end

    # Per-rung: warn once for each rung that just completed (dispatch final, nothing pending)
    # with at least one failed trial. Compared against the counterfactual with this entry still
    # Pending, since telling one entry can retroactively complete rungs above it too (a rung can
    # become dispatch-final via its own hard capacity, independent of the rung below finishing),
    # not just its own -- this comparison catches exactly the ones it just did, exactly once.
    before = [e.id == entry.id ? RunEntry(e.id, e.params, e.metadata; pre_artefact=e.pre_artefact) : e for e in runs]
    for i in 1:k
        _rung_resolved(s, runs, k, i) || continue
        _rung_resolved(s, before, k, i) && continue # already known complete before this tell
        _rung_has_failure(runs, k, i) &&
            @warn "$(typeof(s)): rung $i of bracket $k completed with at least one failed trial"
    end
    return nothing
end
