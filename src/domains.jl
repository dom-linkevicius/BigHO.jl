"""
    Domain

A hyperparameter's kind and candidate values -- nominal, ordinal, or
continuous. `type` (`:nominal`/`:ordinal`/`:continuous`) tags the kind,
`values` holds the sorted candidates, `weights` is an optional per-candidate
sampling weight. Construct via [`Nominal`](@ref)/[`Ordinal`](@ref)/
[`Continuous`](@ref) rather than this struct directly.
"""
struct Domain
    type::Symbol
    values::AbstractVector
    weights::Union{Vector{Float64},Nothing}

    function Domain(type::Symbol, values::AbstractVector, weights::Union{Vector{Float64},Nothing})
        type in (:nominal, :ordinal, :continuous) ||
            throw(ArgumentError("type must be :nominal, :ordinal, or :continuous, got $(repr(type))"))
        weights === nothing || length(weights) == length(values) ||
            throw(ArgumentError("weights must have length $(length(values)) (one per level), got $(length(weights))"))
        weights === nothing || (all(>=(0), weights) && any(>(0), weights)) ||
            throw(ArgumentError("weights must be non-negative, with at least one strictly positive; got $(count(<(0), weights)) negative and $(count(==(0), weights)) zero, out of $(length(weights))"))
        return new(type, values, weights)
    end
end

"""
Default RNG for `rand(d::Domain)` calls with no explicit `rng` argument.
"""
const DEFAULT_DOMAIN_RNG = StableRNG(1)
Base.rand(d::Domain) = rand(DEFAULT_DOMAIN_RNG, d)

# Dispatch on the runtime type of `d.weights` (concretely `Nothing` or
# `Vector{Float64}`, even though the field's static type is a `Union`)
# instead of branching on it -- each method below is fully specialized for
# its case.
_draw(rng::Random.AbstractRNG, d::Domain, ::Nothing) = rand(rng, d.values)
_draw(rng::Random.AbstractRNG, d::Domain, weights::Vector{Float64}) = sample(rng, d.values, Weights(weights))

"""
    rand([rng,] d::Domain)

Draw a candidate from `d`, uniformly unless `d.weights` was given.
"""
Base.rand(rng::Random.AbstractRNG, d::Domain) = _draw(rng, d, d.weights)

"""
    x in d::Domain

Whether `x` is one of `d`'s candidate values. Always a `Bool`, even for
`x === missing`. `d.values` is the exact, already-materialized list of
candidates for every domain kind (never recomputed via arithmetic at check
time), so this is always plain equality -- no floating-point tolerance
involved, or needed.
"""
Base.in(x, d::Domain) = x in d.values
Base.in(::Missing, ::Domain) = false

# Shared construct logic for the standard "levels only" vs "explicit values
# list" constructor pair, common to Nominal and Ordinal. No need to
# separately validate weights here -- Domain's own inner constructor does.
_from_levelcount(type::Symbol, levels::Int, weights) = Domain(type, Base.OneTo(levels), weights)
_from_values(type::Symbol, values::AbstractVector, weights) = Domain(type, collect(values), weights)

"""
    Nominal(levels::Int; weights=nothing)
    Nominal(values::AbstractVector; weights=nothing)

A discrete dimension with no meaningful order between its `levels` values
(e.g. `Nominal([tanh, exp, identity])`). `weights`, if given, biases `rand`
away from uniform.
"""
Nominal(levels::Int; weights::Union{Vector{Float64},Nothing}=nothing) = _from_levelcount(:nominal, levels, weights)
Nominal(values::AbstractVector; weights::Union{Vector{Float64},Nothing}=nothing) = _from_values(:nominal, values, weights)

"""
    Ordinal(levels::Int; weights=nothing)
    Ordinal(values::AbstractVector; weights=nothing)

A discrete dimension with a meaningful order (e.g. `Ordinal(["low",
"medium", "high"])`), even where the values aren't numeric. Numeric
`values` must be sorted increasing (`ArgumentError` otherwise); order can't
be verified for non-numeric values, so construction just warns and trusts
the given order. `weights`, if given, biases `rand` away from uniform.
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
    Continuous(values::AbstractVector{<:Real}; weights=nothing)

A continuous real-valued dimension. The `(min, max, dt)` form is a grid
from `min` to `max` spaced `dt` apart (`dt` is the point spacing, not a
count; the last point falls short of `max` if it doesn't divide evenly).
The `values` form takes an explicit, strictly increasing sequence instead
(e.g. a log-spaced range like `exp10.(LinRange(-1, 3, 50))`) -- unlike the
`(min, max, dt)` form, `values` need not be evenly spaced. `weights`, if
given, biases `rand` away from uniform.
"""
function Continuous(min::Real, max::Real, dt::Real; weights::Union{Vector{Float64},Nothing}=nothing)
    min, max, dt = Float64(min), Float64(max), Float64(dt)
    max >= min || throw(ArgumentError("max ($max) must be >= min ($min)"))
    dt > 0 || throw(ArgumentError("dt must be > 0, got $dt"))
    return Continuous(min:dt:max; weights)
end
function Continuous(values::AbstractVector{<:Real}; weights::Union{Vector{Float64},Nothing}=nothing)
    issorted(values) && allunique(values) ||
        throw(ArgumentError("Continuous(values) requires strictly increasing values (no duplicates); got $values"))
    length(values) < 5 && @warn "Continuous domain has fewer than 5 candidate values; consider using Ordinal instead"
    return Domain(:continuous, Float64.(values), weights)
end
