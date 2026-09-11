"""
    Hyperband(; R, η=3, r_min=1, inner=RandomSampler())
"""
const Hyperband = SuccessiveHalving{true}

# Walks brackets top-down; a bracket is done for good once its top rung resolves, never revisited.
# Pure (never calls `inner`) so exhausted/blocked/create_run_entry can safely re-derive it.
function _bracket_decision(s::Hyperband, k::Int, runs)
    R, r_min, η = s.R, s.r_min, s.η
    _dispatched_count(runs, k, 1) < _capacity(R, r_min, η, k, 1) && return (:draw, k)
    for i in 1:(k-1)
        _rung_resolved(s, runs, k, i) || return (:wait,)
        told = _told_sorted(runs, k, i)
        target = min(_capacity(R, r_min, η, k, i + 1), length(told))
        target == 0 && return _fallback_bracket(s, k, runs)
        n_promoted = _dispatched_count(runs, k, i + 1)
        n_promoted < target && return (:promote, k, i, first(told[n_promoted+1]))
    end
    _rung_resolved(s, runs, k, k) || return (:wait,)
    return _fallback_bracket(s, k, runs)
end

# Whether rung i is done: fully dispatched (possibly shrunk by failures) and nothing Pending.
# Rung 1's target is fixed; i>1's target is capped by how many rung i-1 actually delivered.
function _rung_resolved(s::Hyperband, runs, k::Int, i::Int)
    R, r_min, η = s.R, s.r_min, s.η
    if i == 1
        target = _capacity(R, r_min, η, k, 1)
    else
        _rung_resolved(s, runs, k, i - 1) || return false
        target = min(_capacity(R, r_min, η, k, i), length(_told_sorted(runs, k, i - 1)))
    end
    return _dispatched_count(runs, k, i) >= target && _pending_count(runs, k, i) == 0
end

# Warns exactly once per rung, at the tell! that empties its last Pending entry. Relies on
# _rung_resolved's upstream-first precondition, which only Hyperband's dispatch pattern guarantees.
function on_tell!(s::Hyperband, runs, entry)
    k = entry.metadata[:bracket_k]
    i = entry.metadata[:rung]
    i < k || return nothing # top rung: no promotion decision is ever made from here
    _rung_resolved(s, runs, k, i) || return nothing
    told = _told_sorted(runs, k, i)
    if isempty(told)
        @warn "$(typeof(s)): every trial at rung $i of bracket $k failed -- abandoning bracket $k"
    else
        wanted = _capacity(s.R, s.r_min, s.η, k, i + 1)
        length(told) < wanted && @warn "$(typeof(s)): only $(length(told))/$wanted trials completed at rung $i of bracket $k -- promoting fewer than planned into rung $(i + 1)"
    end
    return nothing
end
