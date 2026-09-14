# Hyperband and ASHA share this exact shape (Sync=true/false picks _bracket_decision). Immutable --
# bracket/rung state is derived fresh from `runs` each time, nothing cached.
struct SuccessiveHalving{Sync,S<:Sampler} <: Sampler
    R::Int
    r_min::Int
    η::Int
    inner::S # draws fresh bottom-rung candidates -- any Sampler, e.g. RandomSampler (default) or LHSampler
end
function SuccessiveHalving{Sync}(; R::Int, η::Int=3, r_min::Int=1, inner::Sampler=RandomSampler()) where {Sync}
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    η > 1 || throw(ArgumentError("η must be greater than 1, got $η"))
    r_min > 0 || throw(ArgumentError("r_min must be positive, got $r_min"))
    r_min <= R || throw(ArgumentError("r_min must be <= R, got r_min=$r_min, R=$R"))
    r_top = r_min * η^_smax(R, r_min, η)
    r_top == R ||
        @warn "SuccessiveHalving: the top resource level reached is $r_top, short of the requested R=$R -- smax=⌊log_η(R/r_min)⌋ floors to the nearest integer, so the schedule only lands exactly on R when R/r_min is an exact power of η"
    return SuccessiveHalving{Sync,typeof(inner)}(R, r_min, η, inner)
end

# :r is stamped onto a trial's params at entry creation, never added to ho.candidates: the resource
# level comes from the schedule, so it has no domain to sample and no unit coordinate to carry.
_add_r(params::NamedTuple, r::Int) = merge((r=r,), params)

# ndigits(n;base) computes ⌊log_base(n)⌋ exactly (no floating-point log, unlike floor(log(R/r_min)/log(η))
# which misrounds on exact powers of η). R÷r_min is safe here since floor is monotonic.
_smax(R::Int, r_min::Int, η::Int) = ndigits(R ÷ r_min; base=η) - 1
# Bracket k starts at r_min*η^(k-1) and always tops out at R: bracket 1 is the full ladder
# (smallest starting budget, most rungs), bracket smax+1 is a single rung at R.
_n_rungs(R::Int, r_min::Int, η::Int, k::Int) = _smax(R, r_min, η) + 2 - k
function _capacity(R::Int, r_min::Int, η::Int, k::Int, i::Int)
    smax = _smax(R, r_min, η)
    n0 = ceil(Int, (smax + 1) * η^(smax + 1 - k) / _n_rungs(R, r_min, η, k))
    return max(1, floor(Int, n0 / η^(i - 1)))
end
_resource(::Int, r_min::Int, η::Int, k::Int, i::Int) = r_min * η^(k + i - 2)
_at_rung(e, k::Int, i::Int) = get(e.metadata, :bracket_k, nothing) == k && get(e.metadata, :rung, nothing) == i
_dispatched_count(runs, k::Int, i::Int) = count(e -> _at_rung(e, k, i), runs)
_promoted_ids(runs, k::Int, i::Int) =
    Set(e.metadata[:promoted_from] for e in runs if _at_rung(e, k, i + 1))
_told_sorted(runs, k::Int, i::Int) =
    sort([(e.id, e.value) for e in runs if e.status === Completed && _at_rung(e, k, i)]; by=last)
_pending_count(runs, k::Int, i::Int) = count(e -> e.status === Pending && _at_rung(e, k, i), runs)
_rung_has_failure(runs, k::Int, i::Int) = any(e -> e.status === Failed && _at_rung(e, k, i), runs)
function _total_trials(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_capacity(R, r_min, η, k, i) for k in 1:(smax+1) for i in 1:_n_rungs(R, r_min, η, k))
end
function _total_draws(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_capacity(R, r_min, η, k, 1) for k in 1:(smax+1))
end

# What _bracket_decision decided. The kind is a type parameter so the consumers dispatch on it;
# :promote is the widest case, so every kind carries its fields and the unused ones are `missing`.
struct SHDecision{K}
    bracket::Union{Int,Missing}
    rung::Union{Int,Missing}
    promoted_from::Union{Int,Missing}
end

_draw(bracket::Int, rung::Int) = SHDecision{:draw}(bracket, rung, missing)
_promote(bracket::Int, rung::Int, promoted_from::Int) = SHDecision{:promote}(bracket, rung, promoted_from)
_wait() = SHDecision{:wait}(missing, missing, missing)
_exhausted() = SHDecision{:exhausted}(missing, missing, missing)

_fallback_bracket(s::SuccessiveHalving, k::Int, runs) =
    k < _smax(s.R, s.r_min, s.η) + 1 ? _bracket_decision(s, k + 1, runs) : _exhausted()

_propose(::SHDecision{:draw}, s::SuccessiveHalving, candidates, runs) = _sample_sh_inner(s, candidates, runs)
_propose(d::SHDecision{:promote}, ::SuccessiveHalving, candidates, runs) = copy(runs[d.promoted_from].unit_params)
_propose(::SHDecision{:wait}, s::SuccessiveHalving, candidates, runs) =
    throw(ArgumentError("$(typeof(s)) has nothing to propose right now -- every bracket is waiting on trials that were asked but not yet told; `blocked` reports this"))
_propose(::SHDecision{:exhausted}, s::SuccessiveHalving, candidates, runs) =
    throw(ArgumentError("$(typeof(s)) has finished its one-pass schedule and can propose nothing further; `exhausted` reports this"))

function _entry_for(d::SHDecision{:draw}, s::SuccessiveHalving, ho, id, params, unit_params)
    with_r = _add_r(params, _resource(s.R, s.r_min, s.η, d.bracket, d.rung))
    return RunEntry(id, with_r, unit_params, Dict{Symbol,Any}(:rung => d.rung, :bracket_k => d.bracket))
end

function _entry_for(d::SHDecision{:promote}, s::SuccessiveHalving, ho, id, params, unit_params)
    with_r = _add_r(params, _resource(s.R, s.r_min, s.η, d.bracket, d.rung + 1))
    metadata = Dict{Symbol,Any}(:rung => d.rung + 1, :bracket_k => d.bracket, :promoted_from => d.promoted_from)
    return RunEntry(id, with_r, unit_params, metadata; pre_artefact=ho.runs[d.promoted_from].post_artefact)
end

function (s::SuccessiveHalving)(candidates, runs)
    return _propose(_bracket_decision(s, 1, runs), s, candidates, runs)
end

_sample_sh_inner(s::SuccessiveHalving, candidates, runs) = _sample_sh_inner(s.inner, candidates, runs)
_sample_sh_inner(inner::Sampler, candidates, runs) = inner(candidates, runs)
# LHSampler is row-indexed off length(runs) -- needs only fresh (never-promoted) draws so its
# row index stays aligned with the design it was built for, not inflated by promotions.
_sample_sh_inner(inner::LHSampler, candidates, runs) = inner(candidates, filter(e -> get(e.metadata, :rung, nothing) == 1, runs))

function init(s::SuccessiveHalving{Sync}, candidates, n) where {Sync}
    inner = init(s.inner, candidates, _total_draws(s.R, s.r_min, s.η))
    return SuccessiveHalving{Sync,typeof(inner)}(s.R, s.r_min, s.η, inner)
end
exhausted(s::SuccessiveHalving, ho) = _bracket_decision(s, 1, ho.runs) isa SHDecision{:exhausted}
blocked(s::SuccessiveHalving, ho) = _bracket_decision(s, 1, ho.runs) isa SHDecision{:wait}

function create_run_entry(s::SuccessiveHalving, ho, id, params, unit_params)
    action = _bracket_decision(s, 1, ho.runs)
    return _entry_for(action, s, ho, id, params, unit_params)
end
