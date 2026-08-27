"""
    Domain

Describes *what kind* of variable a hyperparameter is, and the actual
candidate values it can take: nominal or ordinal (discrete -- see
[`Nominal`](@ref)/[`Ordinal`](@ref)) or continuous (a numeric range
discretized at some resolution -- see [`Continuous`](@ref)). A `Domain` is
the single source of truth for a `Hyperoptimizer` parameter: samplers draw
directly from it via `rand`, no separate candidate array needed.

One concrete representation backs all three kinds: `type` (`:nominal`,
`:ordinal`, or `:continuous`) tags which kind it is, `values` holds the
actual candidates (an index range `1:levels`, an explicit values list, or a
`min:dt:max` grid), and `weights` is an optional non-uniform sampling weight
per candidate -- the hook later machinery (e.g. BOHB) uses to bias sampling
towards promising regions after fitting a density estimate. Construct via
[`Nominal`](@ref)/[`Ordinal`](@ref)/[`Continuous`](@ref) rather than this
struct directly.

The weights/type invariant is enforced here, in `Domain`'s own inner
constructor, rather than only in `Nominal`/`Ordinal`/`Continuous` -- so it
holds no matter how a `Domain` was built. Without this, a wrong-length
`weights` bypassing those three functions wouldn't error at all: StatsBase's
`sample` doesn't validate `length(weights) == length(population)` either, so
a mismatch would silently degrade into always drawing the same element,
forever, rather than throwing.
"""
struct Domain
    type::Symbol
    values::AbstractVector
    weights::Union{Vector{Float64},Nothing}

    function Domain(type::Symbol, values::AbstractVector, weights::Union{Vector{Float64},Nothing})
        type in (:nominal, :ordinal, :continuous) ||
            throw(ArgumentError("type must be :nominal, :ordinal, or :continuous, got $(repr(type))"))
        _check_weights(length(values), weights)
        return new(type, values, weights)
    end
end

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

nlevels(d::Domain) = length(d.values)

# Dispatch on the runtime type of `d.weights` (concretely `Nothing` or
# `Vector{Float64}`, even though the field's static type is a `Union`)
# instead of branching on it -- each method below is fully specialized for
# its case.
_draw(rng::Random.AbstractRNG, d::Domain) = _draw(rng, d, d.weights)
_draw(rng::Random.AbstractRNG, d::Domain, ::Nothing) = rand(rng, d.values)
_draw(rng::Random.AbstractRNG, d::Domain, weights::Vector{Float64}) = sample(rng, d.values, Weights(weights))

"""
    rand([rng,] d::Domain)

Draw a candidate from `d`: a level index, or (if `d` was constructed from a
values list) the corresponding value, for [`Nominal`](@ref)/[`Ordinal`](@ref);
a grid point, for [`Continuous`](@ref). Uniform unless `d.weights` was given.
"""
Base.rand(rng::Random.AbstractRNG, d::Domain) = _draw(rng, d)

"""
    x in d::Domain

Membership test: `x` is a member of `d` if it's one of `d`'s candidate
values (`d.values`). For a [`Continuous`](@ref) domain this uses an
isapprox-tolerant grid check instead of exact equality -- Julia's own range
membership test uses exact floating-point equality, which is fragile when
`dt` doesn't evenly divide `(max - min)`. `missing` is never a member of any
domain -- membership is always a definite `Bool`, never Julia's usual
3-valued `missing`-propagating comparison result.
"""
Base.in(x, d::Domain) = _in(x, d, Val(d.type))
Base.in(::Missing, ::Domain) = false

_in(x, d::Domain, ::Union{Val{:nominal},Val{:ordinal}}) = x in d.values
_in(x, d::Domain, ::Val{:continuous}) = _in_continuous(x, d)

_in_continuous(x, ::Domain) = false # non-Real x can never be a grid point
function _in_continuous(x::Real, d::Domain)
    r = d.values
    (first(r) <= x <= last(r)) || return false
    dt = step(r)
    idx = round(Int, (x - first(r)) / dt)
    return isapprox(x, first(r) + idx * dt; atol=1e-9 * max(1, abs(dt)))
end

# Shared construct logic for the standard "levels only" vs "explicit values
# list" constructor pair, common to Nominal and Ordinal. No need to
# separately validate weights here -- Domain's own inner constructor does.
_from_levelcount(type::Symbol, levels::Int, weights) = Domain(type, Base.OneTo(levels), weights)
_from_values(type::Symbol, values::AbstractVector, weights) = Domain(type, collect(values), weights)

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
Nominal(levels::Int; weights::Union{Vector{Float64},Nothing}=nothing) = _from_levelcount(:nominal, levels, weights)
Nominal(values::AbstractVector; weights::Union{Vector{Float64},Nothing}=nothing) = _from_values(:nominal, values, weights)

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
`Vector{Float64}` of length `levels`) to draw non-uniformly instead.
"""
Ordinal(levels::Int; weights::Union{Vector{Float64},Nothing}=nothing) = _from_levelcount(:ordinal, levels, weights)
function Ordinal(values::AbstractVector; weights::Union{Vector{Float64},Nothing}=nothing)
    _check_order(values)
    return _from_values(:ordinal, values, weights)
end

_check_order(values::AbstractVector{<:Real}) =
    issorted(values) || throw(ArgumentError("Ordinal requires numeric values to be sorted in increasing order; got $values -- use Nominal if order doesn't apply"))
_check_order(values::AbstractVector) =
    @warn "Ordinal cannot verify order for non-numeric values (default isless doesn't reliably match intended domain order); assuming this is the intended order" values

"""
    Continuous(min, max, dt; weights=nothing)

A continuous real-valued dimension bounded by `[min, max]`, discretized into
a grid of points spaced `dt` apart starting at `min` (`dt` is the distance
between neighboring points, not a point count -- the number of points is
derived). If `(max-min)` isn't an exact multiple of `dt`, the last grid
point falls short of `max` rather than overshooting it. Order here is
continuous and enforced by construction (`max >= min` is checked).

`Continuous` only ever represents a *uniformly* spaced grid; for a
non-uniformly spaced range (e.g. log-spaced), use [`Ordinal`](@ref) with the
explicit values instead (e.g. `Ordinal(exp10.(LinRange(-1, 3, 50)))`).

By default (`weights=nothing`), `rand` draws a grid point uniformly; pass
`weights` (a `Vector{Float64}`, one per grid point) to draw non-uniformly
instead.
"""
function Continuous(min::Real, max::Real, dt::Real; weights::Union{Vector{Float64},Nothing}=nothing)
    min, max, dt = Float64(min), Float64(max), Float64(dt)
    max >= min || throw(ArgumentError("max ($max) must be >= min ($min)"))
    dt > 0 || throw(ArgumentError("dt must be > 0, got $dt"))
    return Domain(:continuous, min:dt:max, weights)
end
