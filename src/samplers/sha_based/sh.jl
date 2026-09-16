struct ActiveRung
    rung::Int
    capacity::Int
    resource::Int
    ids::Vector{Int}
end

struct ActiveBracket
    iteration::Int
    bracket::Int
    rungs::Vector{ActiveRung}
end

struct SuccessiveHalving{Sync,S<:Sampler} <: Sampler
    R::Int
    r_min::Int
    η::Int
    iterations::Int
    inner::S
    active_brackets::Vector{ActiveBracket}
    opened::Ref{Tuple{Int,Int}}
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
    return SuccessiveHalving{Sync,typeof(inner)}(R, r_min, η, iterations, inner, ActiveBracket[], Ref((1, 0)))
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

_label(b::ActiveBracket) = "bracket $(b.bracket) of iteration $(b.iteration)"

_dispatched_count(r::ActiveRung) = length(r.ids)
_pending_count(runs, r::ActiveRung) = count(i -> runs[i].status === Pending, r.ids)
_rung_has_failure(runs, r::ActiveRung) = any(i -> runs[i].status === Failed, r.ids)
_told_sorted(runs, r::ActiveRung) =
    sort([(i, runs[i].value) for i in r.ids if runs[i].status === Completed]; by=last)
_promoted_ids(runs, r::ActiveRung) = Set(runs[i].metadata[:promoted_from] for i in r.ids)

function _make_bracket(s::SuccessiveHalving, iteration::Int, bracket::Int)
    rungs = [ActiveRung(i, _capacity(s.R, s.r_min, s.η, bracket, i),
                        _resource(s.R, s.r_min, s.η, bracket, i), Int[])
             for i in 1:_n_rungs(s.R, s.r_min, s.η, bracket)]
    return ActiveBracket(iteration, bracket, rungs)
end

function _open_next!(s::SuccessiveHalving)
    iteration, bracket = s.opened[]
    if bracket < _smax(s.R, s.r_min, s.η) + 1
        bracket += 1
    elseif iteration < s.iterations
        iteration, bracket = iteration + 1, 1
    else
        return nothing
    end
    s.opened[] = (iteration, bracket)
    push!(s.active_brackets, _make_bracket(s, iteration, bracket))
    return last(s.active_brackets)
end

struct SHDecision{K}
    bracket::Union{ActiveBracket,Missing}
    rung::Union{ActiveRung,Missing}
    promoted_from::Union{Int,Missing}
end

_draw(bracket::ActiveBracket, rung::ActiveRung) = SHDecision{:draw}(bracket, rung, missing)
_promote(bracket::ActiveBracket, rung::ActiveRung, promoted_from::Int) =
    SHDecision{:promote}(bracket, rung, promoted_from)
_wait() = SHDecision{:wait}(missing, missing, missing)
_done() = SHDecision{:done}(missing, missing, missing)
_exhausted() = SHDecision{:exhausted}(missing, missing, missing)

function _decide!(s::SuccessiveHalving, runs)
    while true
        isempty(s.active_brackets) && _open_next!(s) === nothing && return _exhausted()
        decision = _bracket_decision(s, first(s.active_brackets), runs)
        decision isa SHDecision{:done} || return decision
        popfirst!(s.active_brackets)
    end
end

_propose(d::SHDecision{:draw}, s::SuccessiveHalving, candidates, runs) = _sample_sh_inner(s, candidates, runs, d)
_propose(d::SHDecision{:promote}, ::SuccessiveHalving, candidates, runs) = copy(runs[d.promoted_from].unit_params)
_propose(::SHDecision{:wait}, s::SuccessiveHalving, candidates, runs) =
    throw(ArgumentError("$(typeof(s)) has nothing to propose right now -- every bracket is waiting on trials that were asked but not yet told; `blocked` reports this"))
_propose(::SHDecision{:exhausted}, s::SuccessiveHalving, candidates, runs) =
    throw(ArgumentError("$(typeof(s)) has finished all $(s.iterations) iterations of its schedule and can propose nothing further; `exhausted` reports this"))

_metadata(b::ActiveBracket, rung::Int) =
    Dict{Symbol,Any}(:iteration => b.iteration, :bracket => b.bracket, :rung => rung)

function _entry_for(d::SHDecision{:draw}, s::SuccessiveHalving, ho, id, params, unit_params)
    return RunEntry(id, _add_r(params, d.rung.resource), unit_params, _metadata(d.bracket, d.rung.rung))
end

function _entry_for(d::SHDecision{:promote}, s::SuccessiveHalving, ho, id, params, unit_params)
    target = d.bracket.rungs[d.rung.rung+1]
    metadata = _metadata(d.bracket, target.rung)
    metadata[:promoted_from] = d.promoted_from
    return RunEntry(id, _add_r(params, target.resource), unit_params, metadata;
                    pre_artefact=ho.runs[d.promoted_from].post_artefact)
end

_record!(d::SHDecision{:draw}, id::Int) = push!(d.rung.ids, id)
_record!(d::SHDecision{:promote}, id::Int) = push!(d.bracket.rungs[d.rung.rung+1].ids, id)

function _bracket_of(s::SuccessiveHalving, entry)
    iteration, bracket = entry.metadata[:iteration], entry.metadata[:bracket]
    idx = findfirst(b -> b.iteration == iteration && b.bracket == bracket, s.active_brackets)
    return idx === nothing ? nothing : s.active_brackets[idx]
end

function (s::SuccessiveHalving)(candidates, runs)
    return _propose(_decide!(s, runs), s, candidates, runs)
end

_sample_sh_inner(s::SuccessiveHalving, candidates, runs, ::SHDecision{:draw}) = _sample_sh_inner(s.inner, candidates, runs)
_sample_sh_inner(inner::Sampler, candidates, runs) = inner(candidates, runs)
_sample_sh_inner(inner::LHSampler, candidates, runs) = inner(candidates, filter(e -> e.metadata[:rung] == 1, runs))

function init(s::SuccessiveHalving{Sync}, candidates, n) where {Sync}
    inner = init(s.inner, candidates, _total_draws(s))
    return SuccessiveHalving{Sync,typeof(inner)}(s.R, s.r_min, s.η, s.iterations, inner,
                                                 ActiveBracket[], Ref((1, 0)))
end
exhausted(s::SuccessiveHalving, ho) = _decide!(s, ho.runs) isa SHDecision{:exhausted}
blocked(s::SuccessiveHalving, ho) = _decide!(s, ho.runs) isa SHDecision{:wait}

function create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    decision = _decide!(s, ho.runs)
    entry = _entry_for(decision, s, ho, id, params, unit_params)
    _record!(decision, id)
    return entry
end
