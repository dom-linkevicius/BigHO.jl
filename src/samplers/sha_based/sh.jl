struct SuccessiveHalving{Sync,S<:Sampler} <: Sampler
    R::Int
    r_min::Int
    η::Int
    iterations::Int
    inner::S
end
function SuccessiveHalving{Sync}(; R::Int, η::Int=3, r_min::Int=1, iterations::Int=1,
                                 inner::Sampler=RandomSampler()) where {Sync}
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    η > 1 || throw(ArgumentError("η must be greater than 1, got $η"))
    r_min > 0 || throw(ArgumentError("r_min must be positive, got $r_min"))
    r_min <= R || throw(ArgumentError("r_min must be <= R, got r_min=$r_min, R=$R"))
    iterations > 0 || throw(ArgumentError("iterations must be positive, got $iterations"))
    r_top = r_min * η^_smax(R, r_min, η)
    r_top == R ||
        @warn "SuccessiveHalving: the top resource level reached is $r_top, short of the requested R=$R -- smax=⌊log_η(R/r_min)⌋ floors to the nearest integer, so the schedule only lands exactly on R when R/r_min is an exact power of η"
    return SuccessiveHalving{Sync,typeof(inner)}(R, r_min, η, iterations, inner)
end

_add_r(params::NamedTuple, r::Int) = merge((r=r,), params)

_smax(R::Int, r_min::Int, η::Int) = ndigits(R ÷ r_min; base=η) - 1
_n_rungs(R::Int, r_min::Int, η::Int, k::Int) = _smax(R, r_min, η) + 2 - k
function _capacity(R::Int, r_min::Int, η::Int, k::Int, i::Int)
    smax = _smax(R, r_min, η)
    n0 = ceil(Int, (smax + 1) * η^(smax + 1 - k) / _n_rungs(R, r_min, η, k))
    return max(1, floor(Int, n0 / η^(i - 1)))
end
_resource(::Int, r_min::Int, η::Int, k::Int, i::Int) = r_min * η^(k + i - 2)

struct BracketId
    iteration::Int
    bracket::Int
end
_label(k::BracketId) = "bracket $(k.bracket) of iteration $(k.iteration)"

_at_rung(e, k::BracketId, i::Int) = e.metadata[:bracket] == k && e.metadata[:rung] == i
_dispatched_count(runs, k::BracketId, i::Int) = count(e -> _at_rung(e, k, i), runs)
_promoted_ids(runs, k::BracketId, i::Int) =
    Set(e.metadata[:promoted_from] for e in runs if _at_rung(e, k, i + 1))
_told_sorted(runs, k::BracketId, i::Int) =
    sort([(e.id, e.value) for e in runs if e.status === Completed && _at_rung(e, k, i)]; by=last)
_pending_count(runs, k::BracketId, i::Int) = count(e -> e.status === Pending && _at_rung(e, k, i), runs)
_rung_has_failure(runs, k::BracketId, i::Int) = any(e -> e.status === Failed && _at_rung(e, k, i), runs)
function _total_trials(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_capacity(R, r_min, η, k, i) for k in 1:(smax+1) for i in 1:_n_rungs(R, r_min, η, k))
end
function _total_draws(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_capacity(R, r_min, η, k, 1) for k in 1:(smax+1))
end
_total_trials(s::SuccessiveHalving) = s.iterations * _total_trials(s.R, s.r_min, s.η)
_total_draws(s::SuccessiveHalving) = s.iterations * _total_draws(s.R, s.r_min, s.η)

struct SHDecision{K}
    bracket_id::Union{BracketId,Missing}
    rung::Union{Int,Missing}
    promoted_from::Union{Int,Missing}
end

_draw(bracket_id::BracketId, rung::Int) = SHDecision{:draw}(bracket_id, rung, missing)
_promote(bracket_id::BracketId, rung::Int, promoted_from::Int) = SHDecision{:promote}(bracket_id, rung, promoted_from)
_wait() = SHDecision{:wait}(missing, missing, missing)
_exhausted() = SHDecision{:exhausted}(missing, missing, missing)

function _fallback_bracket(s::SuccessiveHalving, k::BracketId, runs)
    k.bracket < _smax(s.R, s.r_min, s.η) + 1 && return _bracket_decision(s, BracketId(k.iteration, k.bracket + 1), runs)
    k.iteration < s.iterations && return _bracket_decision(s, BracketId(k.iteration + 1, 1), runs)
    return _exhausted()
end

_propose(::SHDecision{:draw}, s::SuccessiveHalving, candidates, runs) = _sample_sh_inner(s, candidates, runs)
_propose(d::SHDecision{:promote}, ::SuccessiveHalving, candidates, runs) = copy(runs[d.promoted_from].unit_params)
_propose(::SHDecision{:wait}, s::SuccessiveHalving, candidates, runs) =
    throw(ArgumentError("$(typeof(s)) has nothing to propose right now -- every bracket is waiting on trials that were asked but not yet told; `blocked` reports this"))
_propose(::SHDecision{:exhausted}, s::SuccessiveHalving, candidates, runs) =
    throw(ArgumentError("$(typeof(s)) has finished all $(s.iterations) iterations of its schedule and can propose nothing further; `exhausted` reports this"))

function _entry_for(d::SHDecision{:draw}, s::SuccessiveHalving, ho, id, params, unit_params)
    with_r = _add_r(params, _resource(s.R, s.r_min, s.η, d.bracket_id.bracket, d.rung))
    return RunEntry(id, with_r, unit_params, Dict{Symbol,Any}(:rung => d.rung, :bracket => d.bracket_id))
end

function _entry_for(d::SHDecision{:promote}, s::SuccessiveHalving, ho, id, params, unit_params)
    with_r = _add_r(params, _resource(s.R, s.r_min, s.η, d.bracket_id.bracket, d.rung + 1))
    metadata = Dict{Symbol,Any}(:rung => d.rung + 1, :bracket => d.bracket_id, :promoted_from => d.promoted_from)
    return RunEntry(id, with_r, unit_params, metadata; pre_artefact=ho.runs[d.promoted_from].post_artefact)
end

function (s::SuccessiveHalving)(candidates, runs)
    return _propose(_bracket_decision(s, BracketId(1, 1), runs), s, candidates, runs)
end

_sample_sh_inner(s::SuccessiveHalving, candidates, runs) = _sample_sh_inner(s.inner, candidates, runs)
_sample_sh_inner(inner::Sampler, candidates, runs) = inner(candidates, runs)
_sample_sh_inner(inner::LHSampler, candidates, runs) = inner(candidates, filter(e -> e.metadata[:rung] == 1, runs))

function init(s::SuccessiveHalving{Sync}, candidates, n) where {Sync}
    inner = init(s.inner, candidates, _total_draws(s))
    return SuccessiveHalving{Sync,typeof(inner)}(s.R, s.r_min, s.η, s.iterations, inner)
end
exhausted(s::SuccessiveHalving, ho) = _bracket_decision(s, BracketId(1, 1), ho.runs) isa SHDecision{:exhausted}
blocked(s::SuccessiveHalving, ho) = _bracket_decision(s, BracketId(1, 1), ho.runs) isa SHDecision{:wait}

function create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    action = _bracket_decision(s, BracketId(1, 1), ho.runs)
    return _entry_for(action, s, ho, id, params, unit_params)
end
