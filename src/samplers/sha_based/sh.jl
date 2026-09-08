# Hyperband and ASHA share this exact shape (Sync=true/false picks _bracket_decision). Immutable --
# bracket/rung state is derived fresh from `runs` each time, nothing cached.
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
    Sync || !(inner isa TPEWithFallback) ||
        throw(ArgumentError("TPEWithFallback isn't supported with ASHA -- untested against its async promotion rule"))
    r_top = r_min * η^_smax(R, r_min, η)
    r_top == R ||
        @warn "SuccessiveHalving: the top resource level reached is $r_top, short of the requested R=$R -- smax=⌊log_η(R/r_min)⌋ floors to the nearest integer, so the schedule only lands exactly on R when R/r_min is an exact power of η"
    return SuccessiveHalving{Sync}(R, r_min, η, inner)
end

# candidates/params always carry a reserved :r name -- looked up by key rather than assumed to be
# at a fixed position, so removing it never depends on where adding it put it.
_add_r(candidates::NamedTuple, r_domain) = merge(NamedTuple{(:r,)}((r_domain,)), candidates)
_drop_r(params::NamedTuple) = params[filter(!=(:r), keys(params))]
_drop_r(candidates::Tuple) = candidates[2:end] # no keys at this layer -- :r is always first by construction

# ndigits(n;base) computes ⌊log_base(n)⌋ exactly (no floating-point log, unlike floor(log(R/r_min)/log(η))
# which misrounds on exact powers of η). R÷r_min is safe here since floor is monotonic.
_smax(R::Int, r_min::Int, η::Int) = ndigits(R ÷ r_min; base=η) - 1
_resource_levels(R::Int, r_min::Int, η::Int) = [r_min * η^i for i in 0:_smax(R, r_min, η)]
function _capacity(R::Int, r_min::Int, η::Int, k::Int, i::Int)
    smax = _smax(R, r_min, η)
    n0 = ceil(Int, (smax + 1) * η^(k - 1) / k)
    return max(1, floor(Int, n0 / η^(i - 1)))
end
_resource(R::Int, r_min::Int, η::Int, k::Int, i::Int) = r_min * η^(_smax(R, r_min, η) - k + i)
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
    return sum(_capacity(R, r_min, η, k, i) for k in 1:(smax+1) for i in 1:k)
end
function _total_draws(R::Int, r_min::Int, η::Int)
    smax = _smax(R, r_min, η)
    return sum(_capacity(R, r_min, η, k, 1) for k in 1:(smax+1))
end

# Shared across every SuccessiveHalving sampler -- only _sample_sh_inner is dispatched per type
# (a future BOHB would override it instead of delegating to `inner`).
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

# Shared terminal tail-call: fall back to the previous bracket, or declare exhausted.
_fallback_bracket(s::SuccessiveHalving, k::Int, runs) = k > 1 ? _bracket_decision(s, k - 1, runs) : (:exhausted,)

_sample_sh_inner(s::SuccessiveHalving, candidates, runs) = _sample_sh_inner(s.inner, candidates, runs)
_sample_sh_inner(inner::Sampler, candidates, runs) = inner(_drop_r(candidates), runs)
# LHSampler is row-indexed off length(runs) -- needs only fresh (never-promoted) draws so its
# row index stays aligned with the design it was built for, not inflated by promotions.
_sample_sh_inner(inner::LHSampler, candidates, runs) = inner(_drop_r(candidates), filter(e -> get(e.metadata, :rung, nothing) == 1, runs))

init(s::SuccessiveHalving, candidates, n) = typeof(s)(s.R, s.r_min, s.η, init(s.inner, _drop_r(candidates), _total_draws(s.R, s.r_min, s.η)))
exhausted(s::SuccessiveHalving, ho) = first(_bracket_decision(s, _smax(s.R, s.r_min, s.η) + 1, ho.runs)) === :exhausted
blocked(s::SuccessiveHalving, ho) = first(_bracket_decision(s, _smax(s.R, s.r_min, s.η) + 1, ho.runs)) === :wait

function create_run_entry(s::SuccessiveHalving, ho, id, params)
    action = _bracket_decision(s, _smax(s.R, s.r_min, s.η) + 1, ho.runs)
    if action[1] === :draw
        k = action[2]
        return RunEntry(id, params, Dict{Symbol,Any}(:rung => 1, :bracket_k => k))
    end
    k, i, promoted_id = action[2], action[3], action[4]
    metadata = Dict{Symbol,Any}(:rung => i + 1, :bracket_k => k, :promoted_from => promoted_id)
    return RunEntry(id, params, metadata; pre_artefact=ho.runs[promoted_id].post_artefact)
end
