"""
    Domain

A hyperparameter's candidate space: [`Nominal`](@ref), [`Ordinal`](@ref) or [`Continuous`](@ref).
Every domain decodes a `[0,1]` coordinate via [`from_unit`](@ref).
"""
abstract type Domain end

_check_nonempty(values) = isempty(values) && throw(ArgumentError("values must be non-empty"))

"""
    Nominal(levels::Int)
    Nominal(values::AbstractVector)

A discrete dimension with no meaningful order between its values (e.g. `Nominal([tanh, exp, identity])`).
"""
struct Nominal{V<:AbstractVector} <: Domain
    values::V
    function Nominal{V}(values::V) where {V<:AbstractVector}
        _check_nonempty(values)
        return new{V}(values)
    end
end
Nominal(values::AbstractVector) = Nominal{typeof(values)}(values)
Nominal(levels::Int) = Nominal(Base.OneTo(levels))

"""
    Ordinal(levels::Int)
    Ordinal(values::AbstractVector)

A discrete dimension with a meaningful order (e.g. `Ordinal(["low", "medium", "high"])`), even
where the values aren't numeric. Numeric `values` must be sorted increasing (`ArgumentError`
otherwise); order can't be verified for non-numeric values, so construction just warns and trusts
the given order.
"""
struct Ordinal{V<:AbstractVector} <: Domain
    values::V
    function Ordinal{V}(values::V) where {V<:AbstractVector}
        _check_nonempty(values)
        _check_order(values)
        return new{V}(values)
    end
end
Ordinal(values::AbstractVector) = Ordinal{typeof(values)}(values)
Ordinal(levels::Int) = Ordinal(Base.OneTo(levels))

_check_order(values::AbstractVector{<:Real}) =
    issorted(values) || throw(ArgumentError("Ordinal requires numeric values to be sorted in increasing order; got $values -- use Nominal if order doesn't apply"))
_check_order(values::AbstractVector) =
    @warn "Ordinal cannot verify order for non-numeric values (default isless doesn't reliably match intended domain order); assuming this is the intended order" values

"""
    Continuous(min, max; transform=identity)

A continuous dimension over `[min, max]`, with `transform` applied on decode: `Continuous(-4, -1; transform=exp10)` searches log-uniformly but records `0.005`, not `-2.3`.
`transform` need only be finite at both ends of `[min, max]`, which construction checks.
"""
struct Continuous{F} <: Domain
    min::Float64
    max::Float64
    transform::F
    function Continuous{F}(lo::Float64, hi::Float64, transform::F) where {F}
        hi > lo || throw(ArgumentError("max ($hi) must be greater than min ($lo)"))
        _check_finite(transform, lo, hi)
        return new{F}(lo, hi, transform)
    end
end
Continuous(min::Real, max::Real; transform=identity) =
    Continuous{typeof(transform)}(Float64(min), Float64(max), transform)

# Endpoints only: a grid of any size can step over a pole, so sampling the interior wouldn't
# establish anything it doesn't already.
_check_finite(transform, lo::Float64, hi::Float64) =
    (isfinite(transform(lo)) && isfinite(transform(hi))) ||
        throw(ArgumentError("transform must be finite at both ends of [$lo, $hi]; got non-finite values"))

"""
Default RNG for `rand(d::Domain)` calls with no explicit `rng` argument.
"""
const DEFAULT_DOMAIN_RNG = StableRNG(1)

"""
    rand([rng,] d::Domain)

Draw a candidate uniformly from `d` -- over its values for a discrete domain, over `[min, max]`
(before `transform`) for a [`Continuous`](@ref) one.
"""
Base.rand(d::Domain) = rand(DEFAULT_DOMAIN_RNG, d)
Base.rand(rng::Random.AbstractRNG, d::Union{Nominal,Ordinal}) = rand(rng, d.values)
Base.rand(rng::Random.AbstractRNG, d::Continuous) = from_unit(d, rand(rng))

"""
    length(d::Union{Nominal,Ordinal})

How many candidate values the domain has. Undefined for [`Continuous`](@ref), which has no levels.
"""
Base.length(d::Union{Nominal,Ordinal}) = length(d.values)

"""
    from_unit(d::Domain, u::Real)

The candidate at position `u` of `[0,1]`: `u`'s equal-width bin for a discrete domain, `transform(min + u(max - min))` for a [`Continuous`](@ref) one.
"""
from_unit(d::Union{Nominal,Ordinal}, u::Real) = d.values[clamp(ceil(Int, u * length(d)), 1, length(d))]
from_unit(d::Continuous, u::Real) = d.transform(d.min + clamp(u, 0.0, 1.0) * (d.max - d.min))
