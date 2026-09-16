const SHAsync = SuccessiveHalving{false}

"""
    ASHA(; R, η=3, r_min=1, iterations=1, inner=RandomSampler())
"""
const ASHA = SuccessiveHalving{false,<:BasicSamplers}

SuccessiveHalving{false,<:BasicSamplers}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1,
                                         inner::BasicSamplers=RandomSampler()) =
    SuccessiveHalving{false}(; R=R, η=η, r_min=r_min, iterations=iterations, inner=inner)

_n_promotable(s::SHAsync, runs, k::BracketId, i::Int) =
    min(floor(Int, length(_told_sorted(runs, k, i)) / s.η), _capacity(s.R, s.r_min, s.η, k.bracket, i + 1))

function _bracket_decision(s::SHAsync, k::BracketId, runs)
    R, r_min, η = s.R, s.r_min, s.η
    n_rungs = _n_rungs(R, r_min, η, k.bracket)
    for i in (n_rungs-1):-1:1
        promoted = _promoted_ids(runs, k, i)
        if length(promoted) < _n_promotable(s, runs, k, i)
            told = _told_sorted(runs, k, i)
            id, _ = first(t for t in told if first(t) ∉ promoted)
            return _promote(k, i, id)
        end
    end
    _dispatched_count(runs, k, 1) < _capacity(R, r_min, η, k.bracket, 1) && return _draw(k, 1)
    any(i -> _pending_count(runs, k, i) > 0, 1:n_rungs) && return _wait()
    return _fallback_bracket(s, k, runs)
end

function _bracket_has_room(s::SHAsync, k::BracketId, runs)
    _dispatched_count(runs, k, 1) < _capacity(s.R, s.r_min, s.η, k.bracket, 1) && return true
    return any(i -> length(_promoted_ids(runs, k, i)) < _n_promotable(s, runs, k, i), 1:(_n_rungs(s.R, s.r_min, s.η, k.bracket)-1))
end

function _rung_resolved(s::SHAsync, runs, k::BracketId, i::Int)
    dispatch_final = if _dispatched_count(runs, k, i) >= _capacity(s.R, s.r_min, s.η, k.bracket, i)
        true
    elseif i == 1
        false
    else
        _rung_resolved(s, runs, k, i - 1) && _dispatched_count(runs, k, i) >= _n_promotable(s, runs, k, i - 1)
    end
    return dispatch_final && _pending_count(runs, k, i) == 0
end

function on_tell!(s::SHAsync, runs, entry)
    k = entry.metadata[:bracket]
    R, r_min, η = s.R, s.r_min, s.η
    n_rungs = _n_rungs(R, r_min, η, k.bracket)

    if all(i -> _pending_count(runs, k, i) == 0, 1:n_rungs) && !_bracket_has_room(s, k, runs)
        total_capacity = sum(_capacity(R, r_min, η, k.bracket, i) for i in 1:n_rungs)
        total_dispatched = sum(_dispatched_count(runs, k, i) for i in 1:n_rungs)
        total_dispatched < total_capacity && @warn "$(typeof(s)): $(_label(k)) stalled at $total_dispatched/$total_capacity trials dispatched -- no rung can accept more"
    end

    resolved_before = false
    for i in entry.metadata[:rung]:n_rungs
        if _rung_resolved(s, runs, k, i)
            resolved_before || _rung_has_failure(runs, k, i) && @warn "$(typeof(s)): rung $i of $(_label(k)) completed with at least one failed trial"
        end
        i == n_rungs && break
        resolved_before = _dispatched_count(runs, k, i + 1) >= _capacity(R, r_min, η, k.bracket, i + 1) ||
                          (resolved_before && _dispatched_count(runs, k, i + 1) >= _n_promotable(s, runs, k, i))
    end
    return nothing
end
