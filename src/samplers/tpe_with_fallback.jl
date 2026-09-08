"""
    TPEWithFallback(; default=RandomSampler(), top_n_percent=15, num_samples=64, random_fraction=1/3,
                     bandwidth_factor=3, min_points_in_model=nothing, rng=StableRNG(1))

A TPE-style adaptive sampler (per Falkner et al. 2018's BOHB): splits told configs at a given
budget into good/bad kernel density estimates and samples toward the top `1/η`-style ratio
`l(x)/g(x)`, falling back to `default` (plain random by default) until enough data exists to fit
a model, and with probability `random_fraction` even once one can be fit.
Used as `Hyperband(...; inner=TPEWithFallback(...))` -- that combination is what's commonly called
BOHB; not supported with ASHA (its async promotion isn't tested against this).
Pools told configs by resource level (`:r`) and models the highest level with `min_points_in_model`
observations, per Falkner et al. 2018 -- so no special-cased `_sample_sh_inner` override is needed;
the generic `inner::Sampler` dispatch (passing all of `runs`) is enough.

NOTE: first-draft sketch -- per-dimension KDEs (not BOHB's joint multivariate one), so cross-parameter
interactions aren't modeled.
"""
struct TPEWithFallback{T<:Random.AbstractRNG} <: Sampler
    default::Sampler
    top_n_percent::Int
    num_samples::Int
    random_fraction::Float64
    bandwidth_factor::Float64
    min_points_in_model::Union{Int,Nothing}
    rng::T
end
function TPEWithFallback(; default::Sampler=RandomSampler(), top_n_percent::Int=15, num_samples::Int=64,
                          random_fraction::Real=1 / 3, bandwidth_factor::Real=3, min_points_in_model::Union{Int,Nothing}=nothing,
                          rng::Random.AbstractRNG=StableRNG(1))
    0 < top_n_percent < 100 || throw(ArgumentError("top_n_percent must be in (0,100), got $top_n_percent"))
    num_samples > 0 || throw(ArgumentError("num_samples must be positive, got $num_samples"))
    0 <= random_fraction <= 1 || throw(ArgumentError("random_fraction must be in [0,1], got $random_fraction"))
    bandwidth_factor > 0 || throw(ArgumentError("bandwidth_factor must be positive, got $bandwidth_factor"))
    min_points_in_model === nothing || min_points_in_model > 0 ||
        throw(ArgumentError("min_points_in_model must be positive, got $min_points_in_model"))
    return TPEWithFallback(default, top_n_percent, num_samples, Float64(random_fraction), Float64(bandwidth_factor), min_points_in_model, rng)
end

# Silverman's rule of thumb; falls back to a wide default when every value coincides (std is 0/NaN).
_bandwidth(vals::AbstractVector{<:Real}) = (s = std(vals); s > 0 ? 1.06 * s * length(vals)^(-1 / 5) : (maximum(vals) - minimum(vals) + 1) / 10)

_is_continuous(d::Domain) = d.type in (:continuous_linear, :continuous_arbitrary)

# Mixture-of-Gaussians density (continuous) or Laplace-smoothed empirical frequency (discrete), fit
# on `vals` -- the same estimator used both to propose candidates and to score them for l(x)/g(x).
function _density(d::Domain, vals, x, widen::Real=1.0)
    _is_continuous(d) || return (count(==(x), vals) + 1) / (length(vals) + length(d.values))
    bw = _bandwidth(vals) * widen
    return sum(exp(-((x - v)^2) / (2bw^2)) / (bw * sqrt(2π)) for v in vals) / length(vals)
end

# Draw one dimension's value from `vals`'s density (see `_density`), snapped/restricted to `d`'s grid.
function _sample_dim(d::Domain, vals, rng, widen::Real=1.0)
    if _is_continuous(d)
        x = rand(rng, vals) + _bandwidth(vals) * widen * randn(rng)
        return d.values[argmin(abs.(d.values .- x))]
    end
    weights = Float64[count(==(v), vals) + 1 for v in d.values]
    return sample(rng, d.values, Weights(weights))
end

_n_good(n::Int, top_n_percent::Int) = max(1, ceil(Int, top_n_percent / 100 * n))

# Best of `num_samples` proposals drawn from the widened good-KDE `l`, scored by log l(x) - log g(x).
function _tpe_draw(s::TPEWithFallback, candidates, told, rng)
    sorted = sort(told; by=e -> e.value)
    n_good = _n_good(length(sorted), s.top_n_percent)
    good = [collect(_drop_r(e.params)) for e in sorted[1:n_good]]
    bad = [collect(_drop_r(e.params)) for e in sorted[(n_good+1):end]]

    best, best_score = nothing, -Inf
    for _ in 1:s.num_samples
        x = [_sample_dim(d, [g[i] for g in good], rng, s.bandwidth_factor) for (i, d) in enumerate(candidates)]
        score = sum(log(_density(d, [g[i] for g in good], x[i], s.bandwidth_factor)) - log(_density(d, [b[i] for b in bad], x[i]))
                     for (i, d) in enumerate(candidates))
        score > best_score && ((best, best_score) = (x, score))
    end
    return best
end

function (s::TPEWithFallback)(candidates, runs)
    told = [e for e in runs if e.status === Completed]
    n_min = something(s.min_points_in_model, length(candidates) + 1)
    by_budget = Dict{Any,Vector{RunEntry}}()
    for e in told
        push!(get!(() -> RunEntry[], by_budget, e.params.r), e)
    end
    usable = [r for (r, es) in by_budget if length(es) >= n_min && _n_good(length(es), s.top_n_percent) < length(es)]
    (isempty(usable) || rand(s.rng) < s.random_fraction) && return s.default(candidates, runs)
    return _tpe_draw(s, candidates, by_budget[maximum(usable)], s.rng)
end

on_tell!(::TPEWithFallback, runs, entry) = nothing
init(s::TPEWithFallback, candidates, n) = s
exhausted(::TPEWithFallback, ho) = false
blocked(::TPEWithFallback, ho) = false
create_run_entry(::TPEWithFallback, ho, id, params) = RunEntry(id, params)
