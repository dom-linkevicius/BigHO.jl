"""
    Domain

Describes *what kind* of variable a hyperparameter is, and (optionally) the
actual candidate values it can take -- ordinal (ordered discrete, including
continuous ranges discretized at some resolution) or categorical
(unordered/nominal discrete). A `Domain` is the single source of truth for a
`Hyperoptimizer` parameter: samplers draw directly from it via `rand`, no
separate candidate array needed.
"""
abstract type Domain end

"""
    Ordinal <: Domain

Any domain with an inherent order, represented internally as `n` discretized,
weightable positions `1:n` -- a plain enumerated ordinal ([`Levels`](@ref)) or
a continuous range discretized at some resolution ([`Continuous`](@ref)).
Sharing this representation is deliberate: later machinery (e.g. BOHB
updating sampling weights from a fitted KDE) can be written once against
`Ordinal` and apply to both.
"""
abstract type Ordinal <: Domain end

"""
Shared default RNG for `rand(d::Domain)` calls with no explicit `rng`
argument. A persistent `StableRNG` (not `Random.default_rng()`), for the same
reason `RandomSampler()` defaults to `StableRNG` rather than
`MersenneTwister`: a reproducible, version-stable stream rather than
whatever Julia's global RNG happens to be. It's a single shared, mutating
instance (unlike `RandomSampler`, which gets its own fresh `StableRNG(1)` per
instance) because repeated `rand(d)` calls need to actually advance -- there
is no per-call object to hold that state in.
"""
const DEFAULT_DOMAIN_RNG = StableRNG(1)

_check_weights(levels::Int, weights::Nothing) = nothing
function _check_weights(levels::Int, weights::Vector{Float64})
    length(weights) == levels || throw(ArgumentError("weights must have length $levels (one per level), got $(length(weights))"))
    return nothing
end

# Shared sampling mechanics for any domain with `nlevels(d)` discretized
# positions `1:n` and an optional `d.weights` over them (`nothing` => uniform).
# `nlevels` methods for each concrete type are defined alongside that type,
# below, since they need the struct to already exist.
nlevels(d::Ordinal) = d.levels

function _sample_level(rng::Random.AbstractRNG, d::Domain)
    n = nlevels(d)
    return d.weights === nothing ? rand(rng, 1:n) : sample(rng, 1:n, Weights(d.weights))
end

"""
    Levels(levels::Int; weights=nothing)
    Levels(values::AbstractVector; weights=nothing)

A plain enumerated ordinal dimension with `levels` distinct values that have
a meaningful order (e.g. "low" < "medium" < "high") even though the values
themselves need not be numeric.

Construct from a plain level count (`rand` then returns a level index, an
`Int` in `1:levels`), or from the actual candidate values directly (e.g.
`Levels([true, false])`, replacing what used to be a plain candidate array --
`rand` then returns one of those values).

By default (`weights=nothing`), `rand` draws uniformly; pass `weights` (a
`Vector{Float64}` of length `levels`) to draw non-uniformly instead.
"""
struct Levels <: Ordinal
    levels::Int
    values::Union{Vector,Nothing}
    weights::Union{Vector{Float64},Nothing}
end
function Levels(levels::Int; weights::Union{Vector{Float64},Nothing}=nothing)
    _check_weights(levels, weights)
    return Levels(levels, nothing, weights)
end
function Levels(values::AbstractVector; weights::Union{Vector{Float64},Nothing}=nothing)
    _check_weights(length(values), weights)
    return Levels(length(values), collect(values), weights)
end

"""
    rand([rng,] d::Levels)

Draw a level index from `1:d.levels`, uniformly unless `d.weights` was
given -- or, if `d` was constructed from a values list, the corresponding
value from that list.
"""
function Base.rand(rng::Random.AbstractRNG, d::Levels)
    idx = _sample_level(rng, d)
    return d.values === nothing ? idx : d.values[idx]
end
Base.rand(d::Levels) = rand(DEFAULT_DOMAIN_RNG, d)

Base.in(x, d::Levels) = d.values === nothing ? (x isa Integer && 1 <= x <= d.levels) : (x in d.values)

"""
    Continuous(min, max, dt; weights=nothing)

A continuous real-valued dimension bounded by `[min, max]`, discretized into
a grid of points spaced `dt` apart starting at `min` (`dt` is the distance
between neighboring points, not a point count -- the number of points,
`floor((max-min)/dt)+1`, is derived). If `(max-min)` isn't an exact multiple
of `dt`, the last grid point falls short of `max` rather than overshooting it.

`Continuous` only ever represents a *uniformly* spaced grid; for a
non-uniformly spaced range (e.g. log-spaced), use [`Levels`](@ref) with the
explicit values instead (e.g. `Levels(exp10.(LinRange(-1, 3, 50)))`).

By default (`weights=nothing`), `rand` draws a grid point uniformly; pass
`weights` (a `Vector{Float64}`, one per grid point) to draw non-uniformly
instead -- this is the hook later machinery (e.g. BOHB) uses to bias sampling
towards promising regions after fitting a density estimate.
"""
struct Continuous <: Ordinal
    min::Float64
    max::Float64
    dt::Float64
    weights::Union{Vector{Float64},Nothing}
end
function Continuous(min::Real, max::Real, dt::Real; weights::Union{Vector{Float64},Nothing}=nothing)
    min, max, dt = Float64(min), Float64(max), Float64(dt)
    max >= min || throw(ArgumentError("max ($max) must be >= min ($min)"))
    dt > 0 || throw(ArgumentError("dt must be > 0, got $dt"))
    _check_weights(floor(Int, (max - min) / dt) + 1, weights)
    return Continuous(min, max, dt, weights)
end

nlevels(d::Continuous) = floor(Int, (d.max - d.min) / d.dt) + 1

"""
    rand([rng,] d::Continuous) -> Float64

Draw a grid point from `d`'s discretization, uniformly unless `d.weights`
was given.
"""
function Base.rand(rng::Random.AbstractRNG, d::Continuous)
    idx = _sample_level(rng, d)
    return d.min + (idx - 1) * d.dt
end
Base.rand(d::Continuous) = rand(DEFAULT_DOMAIN_RNG, d)

function Base.in(x, d::Continuous)
    x isa Real || return false
    (d.min <= x <= d.max) || return false
    idx = round(Int, (x - d.min) / d.dt)
    return isapprox(x, d.min + idx * d.dt; atol=1e-9 * max(1, abs(d.dt)))
end

"""
    Categorical(levels::Int; weights=nothing)
    Categorical(values::AbstractVector; weights=nothing)

A discrete dimension with `levels` distinct nominal values that have no
meaningful order (e.g. a choice of activation function). Any two distinct
levels must be treated as equally "different" by design/KDE methods -- unlike
[`Ordinal`](@ref), this is not part of that hierarchy.

Construct from a plain level count (`rand` then returns a level index, an
`Int` in `1:levels`), or from the actual candidate values directly (e.g.
`Categorical([tanh, exp, identity])`, replacing what used to be a plain
candidate array -- `rand` then returns one of those values).

By default (`weights=nothing`), `rand` draws uniformly; pass `weights` (a
`Vector{Float64}` of length `levels`) to draw non-uniformly instead.
"""
struct Categorical <: Domain
    levels::Int
    values::Union{Vector,Nothing}
    weights::Union{Vector{Float64},Nothing}
end
function Categorical(levels::Int; weights::Union{Vector{Float64},Nothing}=nothing)
    _check_weights(levels, weights)
    return Categorical(levels, nothing, weights)
end
function Categorical(values::AbstractVector; weights::Union{Vector{Float64},Nothing}=nothing)
    _check_weights(length(values), weights)
    return Categorical(length(values), collect(values), weights)
end

nlevels(d::Categorical) = d.levels

"""
    rand([rng,] d::Categorical)

Draw a level index from `1:d.levels`, uniformly unless `d.weights` was
given -- or, if `d` was constructed from a values list, the corresponding
value from that list.
"""
function Base.rand(rng::Random.AbstractRNG, d::Categorical)
    idx = _sample_level(rng, d)
    return d.values === nothing ? idx : d.values[idx]
end
Base.rand(d::Categorical) = rand(DEFAULT_DOMAIN_RNG, d)

Base.in(x, d::Categorical) = d.values === nothing ? (x isa Integer && 1 <= x <= d.levels) : (x in d.values)
