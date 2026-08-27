"""
    Domain

Describes *what kind* of variable a hyperparameter is, and (optionally) the
actual candidate values it can take -- categorical (discrete, nominal or
ordinal -- see [`Categorical`](@ref)) or continuous (a numeric range
discretized at some resolution -- see [`Continuous`](@ref)). A `Domain` is
the single source of truth for a `Hyperoptimizer` parameter: samplers draw
directly from it via `rand`, no separate candidate array needed.
"""
abstract type Domain end

"""
    Categorical <: Domain

Any discrete domain, in the classical statistics sense that splits into two
concrete kinds: [`Nominal`](@ref) (no meaningful order between the levels)
and [`Ordinal`](@ref) (a meaningful order, checked where it can be). Both
are represented internally as `n` discretized, weightable positions `1:n`,
sharing that representation with [`Continuous`](@ref) (a `Domain`, not a
`Categorical` -- order there is continuous and enforced by construction
instead) via `Domain`-level sampling machinery. Sharing it is deliberate:
later machinery (e.g. BOHB updating sampling weights from a fitted KDE) can
be written once and apply to all three.
"""
abstract type Categorical <: Domain end

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
Base.rand(d::Domain) = rand(DEFAULT_DOMAIN_RNG, d)

_check_weights(levels::Int, weights::Nothing) = nothing
function _check_weights(levels::Int, weights::Vector{Float64})
    length(weights) == levels || throw(ArgumentError("weights must have length $levels (one per level), got $(length(weights))"))
    return nothing
end

# Shared sampling mechanics for any domain with `nlevels(d)` discretized
# positions `1:n` and an optional `d.weights` over them (`nothing` => uniform).
# `nlevels` methods for each concrete type are defined alongside that type,
# below, since they need the struct to already exist.
nlevels(d::Categorical) = d.levels

# Dispatch on the runtime type of `d.weights` (concretely `Nothing` or
# `Vector{Float64}`, even though the field's static type is a `Union`)
# instead of branching on it -- each method below is fully specialized for
# its case.
_sample_level(rng::Random.AbstractRNG, d::Domain) = _sample_level(rng, d, d.weights)
_sample_level(rng::Random.AbstractRNG, d::Domain, ::Nothing) = rand(rng, 1:nlevels(d))
_sample_level(rng::Random.AbstractRNG, d::Domain, weights::Vector{Float64}) = sample(rng, 1:nlevels(d), Weights(weights))

# Same idiom for `Nominal`/`Ordinal`'s `values` field (`Nothing` or
# `Vector`): dispatch on its runtime type instead of branching on it. Shared
# by both via the `Categorical` ancestor, since they use the identical
# "return the index itself, or the corresponding value if `values` was
# given" representation.
_resolve_value(::Nothing, idx::Int) = idx
_resolve_value(values::Vector, idx::Int) = values[idx]

_value_in(x::Integer, ::Nothing, levels::Int) = 1 <= x <= levels
_value_in(x, ::Nothing, levels::Int) = false # non-Integer x can never be a member of an index-only domain
_value_in(x, values::Vector, ::Int) = x in values

"""
    rand([rng,] d::Categorical)

Draw a level index from `1:d.levels`, uniformly unless `d.weights` was
given -- or, if `d` was constructed from a values list, the corresponding
value from that list. Shared by both [`Nominal`](@ref) and [`Ordinal`](@ref),
which use the identical representation and differ only in whether
construction checks the values' order.
"""
Base.rand(rng::Random.AbstractRNG, d::Categorical) = _resolve_value(d.values, _sample_level(rng, d))

"""
    x in d::Categorical

Membership test: `x` is a member of `d` if `d` was constructed from a values
list and `x` is one of them, or (for a level-count-only `d`) if `x` is an
`Int` in `1:d.levels`.
"""
Base.in(x, d::Categorical) = _value_in(x, d.values, d.levels)

# Shared validate+construct logic for any concrete Categorical subtype with
# the standard (levels, values, weights) field layout -- both constructor
# forms (bare count vs. explicit values) exist for both Nominal and Ordinal,
# differing only in the type being built and (for Ordinal's values form) an
# extra order check layered on top by that type's own constructor.
function _from_levelcount(::Type{T}, levels::Int, weights) where {T<:Categorical}
    _check_weights(levels, weights)
    return T(levels, nothing, weights)
end
function _from_values(::Type{T}, values::AbstractVector, weights) where {T<:Categorical}
    _check_weights(length(values), weights)
    return T(length(values), collect(values), weights)
end

"""
    Nominal(levels::Int; weights=nothing)
    Nominal(values::AbstractVector; weights=nothing)

A discrete dimension with `levels` distinct nominal values that have no
meaningful order (e.g. a choice of activation function) -- any two distinct
levels must be treated as equally "different" by design/KDE methods. For a
domain where order *does* matter, see [`Ordinal`](@ref).

Construct from a plain level count (`rand` then returns a level index, an
`Int` in `1:levels`), or from the actual candidate values directly (e.g.
`Nominal([tanh, exp, identity])`, replacing what used to be a plain
candidate array -- `rand` then returns one of those values). No ordering
check is performed on the values -- that's the whole point of `Nominal`
versus `Ordinal`.

By default (`weights=nothing`), `rand` draws uniformly; pass `weights` (a
`Vector{Float64}` of length `levels`) to draw non-uniformly instead.
"""
struct Nominal <: Categorical
    levels::Int
    values::Union{Vector,Nothing}
    weights::Union{Vector{Float64},Nothing}
end
Nominal(levels::Int; weights::Union{Vector{Float64},Nothing}=nothing) = _from_levelcount(Nominal, levels, weights)
Nominal(values::AbstractVector; weights::Union{Vector{Float64},Nothing}=nothing) = _from_values(Nominal, values, weights)

"""
    Ordinal(levels::Int; weights=nothing)
    Ordinal(values::AbstractVector; weights=nothing)

A discrete dimension with `levels` distinct values that have a meaningful
order (e.g. "low" < "medium" < "high") even though the values themselves
need not be numeric. For a domain where order does *not* apply, see
[`Nominal`](@ref).

Construct from a plain level count (`rand` then returns a level index, an
`Int` in `1:levels` -- there's nothing to check order of, the index space
`1:levels` is ordered by definition), or from the actual candidate values
directly (e.g. `Ordinal([1, 2, 5, 10])`).

Ordering of an explicit values list is checked only when it's actually
verifiable: if `eltype(values) <: Real`, construction requires
`issorted(values)` and throws `ArgumentError` otherwise (Julia's default
`isless` reliably matches numeric intent). For any other element type
(strings, symbols, ...), `isless`/alphabetical order generally does *not*
match the intended domain order (`issorted(["low","medium","high"])` is
`false` -- alphabetically "high" < "low" < "medium" -- even though that's
the canonical example of an intentionally-ordered non-numeric `Ordinal`), so
there's no reliable way to verify it: construction instead emits `@warn`
showing the values in the order given, and assumes that order is intentional.

By default (`weights=nothing`), `rand` draws uniformly; pass `weights` (a
`Vector{Float64}` of length `levels`) to draw non-uniformly instead -- this
is the hook later machinery (e.g. BOHB) uses to bias sampling towards
promising regions after fitting a density estimate.
"""
struct Ordinal <: Categorical
    levels::Int
    values::Union{Vector,Nothing}
    weights::Union{Vector{Float64},Nothing}
end
Ordinal(levels::Int; weights::Union{Vector{Float64},Nothing}=nothing) = _from_levelcount(Ordinal, levels, weights)
function Ordinal(values::AbstractVector; weights::Union{Vector{Float64},Nothing}=nothing)
    _check_order(values)
    return _from_values(Ordinal, values, weights)
end

_check_order(values::AbstractVector{<:Real}) =
    issorted(values) || throw(ArgumentError("Ordinal requires numeric values to be sorted in increasing order; got $values -- use Nominal if order doesn't apply"))
_check_order(values::AbstractVector) =
    @warn "Ordinal cannot verify order for non-numeric values (default isless doesn't reliably match intended domain order); assuming this is the intended order" values

"""
    Continuous(min, max, dt; weights=nothing)

A continuous real-valued dimension bounded by `[min, max]`, discretized into
a grid of points spaced `dt` apart starting at `min` (`dt` is the distance
between neighboring points, not a point count -- the number of points,
`floor((max-min)/dt)+1`, is derived). If `(max-min)` isn't an exact multiple
of `dt`, the last grid point falls short of `max` rather than overshooting it.
Order here is continuous and enforced by construction (`max >= min` is
checked), unlike `Categorical`'s subtypes -- so `Continuous` is a `Domain`
directly, not a `Categorical`.

`Continuous` only ever represents a *uniformly* spaced grid; for a
non-uniformly spaced range (e.g. log-spaced), use [`Ordinal`](@ref) with the
explicit values instead (e.g. `Ordinal(exp10.(LinRange(-1, 3, 50)))`).

By default (`weights=nothing`), `rand` draws a grid point uniformly; pass
`weights` (a `Vector{Float64}`, one per grid point) to draw non-uniformly
instead -- this is the hook later machinery (e.g. BOHB) uses to bias sampling
towards promising regions after fitting a density estimate.
"""
struct Continuous <: Domain
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

Base.in(x, d::Continuous) = false # non-Real x can never be a grid point
function Base.in(x::Real, d::Continuous)
    (d.min <= x <= d.max) || return false
    idx = round(Int, (x - d.min) / d.dt)
    return isapprox(x, d.min + idx * d.dt; atol=1e-9 * max(1, abs(d.dt)))
end
