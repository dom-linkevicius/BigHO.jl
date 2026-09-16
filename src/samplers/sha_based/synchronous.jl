const SHSync = SuccessiveHalving{true}

"""
    Hyperband(; R, η=3, r_min=1, iterations=1, inner=RandomSampler())
"""
const Hyperband = SuccessiveHalving{true,<:BasicSamplers}

SuccessiveHalving{true,<:BasicSamplers}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1,
                                        inner::BasicSamplers=RandomSampler()) =
    SuccessiveHalving{true}(; R=R, η=η, r_min=r_min, iterations=iterations, inner=inner)

function _bracket_decision(s::SHSync, b::ActiveBracket, runs)
    n_rungs = length(b.rungs)
    _dispatched_count(b.rungs[1]) < b.rungs[1].capacity && return _draw(b, b.rungs[1])
    for i in 1:(n_rungs-1)
        _rung_resolved(s, runs, b, i) || return _wait()
        told = _told_sorted(runs, b.rungs[i])
        target = min(b.rungs[i+1].capacity, length(told))
        target == 0 && return _done()
        n_promoted = _dispatched_count(b.rungs[i+1])
        n_promoted < target && return _promote(b, b.rungs[i], first(told[n_promoted+1]))
    end
    _rung_resolved(s, runs, b, n_rungs) || return _wait()
    return _done()
end

function _rung_resolved(s::SHSync, runs, b::ActiveBracket, i::Int)
    if i == 1
        target = b.rungs[1].capacity
    else
        _rung_resolved(s, runs, b, i - 1) || return false
        target = min(b.rungs[i].capacity, length(_told_sorted(runs, b.rungs[i-1])))
    end
    return _dispatched_count(b.rungs[i]) >= target && _pending_count(runs, b.rungs[i]) == 0
end

function on_tell!(s::SHSync, runs, entry)
    b = _bracket_of(s, entry)
    b === nothing && return nothing
    i = entry.metadata[:rung]
    i < length(b.rungs) || return nothing
    _rung_resolved(s, runs, b, i) || return nothing
    told = _told_sorted(runs, b.rungs[i])
    if isempty(told)
        @warn "$(typeof(s)): every trial at rung $i of $(_label(b)) failed -- abandoning it"
    else
        wanted = b.rungs[i+1].capacity
        length(told) < wanted && @warn "$(typeof(s)): only $(length(told))/$wanted trials completed at rung $i of $(_label(b)) -- promoting fewer than planned into rung $(i + 1)"
    end
    return nothing
end
