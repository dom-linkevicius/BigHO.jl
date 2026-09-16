const SHAsync = SuccessiveHalving{false}

"""
    ASHA(; R, η=3, r_min=1, iterations=1, inner=RandomSampler())
"""
const ASHA = SuccessiveHalving{false,<:BasicSamplers}

SuccessiveHalving{false,<:BasicSamplers}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1,
                                         inner::BasicSamplers=RandomSampler()) =
    SuccessiveHalving{false}(; R=R, η=η, r_min=r_min, iterations=iterations, inner=inner)

_n_promotable(s::SHAsync, runs, b::ActiveBracket, i::Int) =
    min(floor(Int, length(_told_sorted(runs, b.rungs[i])) / s.η), b.rungs[i+1].capacity)

function _bracket_decision(s::SHAsync, b::ActiveBracket, runs)
    n_rungs = length(b.rungs)
    for i in (n_rungs-1):-1:1
        promoted = _promoted_ids(runs, b.rungs[i+1])
        if length(promoted) < _n_promotable(s, runs, b, i)
            told = _told_sorted(runs, b.rungs[i])
            id, _ = first(t for t in told if first(t) ∉ promoted)
            return _promote(b, b.rungs[i], id)
        end
    end
    _dispatched_count(b.rungs[1]) < b.rungs[1].capacity && return _draw(b, b.rungs[1])
    any(r -> _pending_count(runs, r) > 0, b.rungs) && return _wait()
    return _done()
end

function _bracket_has_room(s::SHAsync, b::ActiveBracket, runs)
    _dispatched_count(b.rungs[1]) < b.rungs[1].capacity && return true
    return any(i -> length(_promoted_ids(runs, b.rungs[i+1])) < _n_promotable(s, runs, b, i), 1:(length(b.rungs)-1))
end

function _rung_resolved(s::SHAsync, runs, b::ActiveBracket, i::Int)
    dispatch_final = if _dispatched_count(b.rungs[i]) >= b.rungs[i].capacity
        true
    elseif i == 1
        false
    else
        _rung_resolved(s, runs, b, i - 1) && _dispatched_count(b.rungs[i]) >= _n_promotable(s, runs, b, i - 1)
    end
    return dispatch_final && _pending_count(runs, b.rungs[i]) == 0
end

function on_tell!(s::SHAsync, runs, entry)
    b = _bracket_of(s, entry)
    b === nothing && return nothing
    n_rungs = length(b.rungs)

    if all(r -> _pending_count(runs, r) == 0, b.rungs) && !_bracket_has_room(s, b, runs)
        total_capacity = sum(r.capacity for r in b.rungs)
        total_dispatched = sum(_dispatched_count(r) for r in b.rungs)
        total_dispatched < total_capacity && @warn "$(typeof(s)): $(_label(b)) stalled at $total_dispatched/$total_capacity trials dispatched -- no rung can accept more"
    end

    resolved_before = false
    for i in entry.metadata[:rung]:n_rungs
        if _rung_resolved(s, runs, b, i)
            resolved_before || _rung_has_failure(runs, b.rungs[i]) && @warn "$(typeof(s)): rung $i of $(_label(b)) completed with at least one failed trial"
        end
        i == n_rungs && break
        resolved_before = _dispatched_count(b.rungs[i+1]) >= b.rungs[i+1].capacity ||
                          (resolved_before && _dispatched_count(b.rungs[i+1]) >= _n_promotable(s, runs, b, i))
    end
    return nothing
end
