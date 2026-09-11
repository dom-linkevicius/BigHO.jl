"""
    Domain
"""
abstract type Domain end

"""
    Nominal(levels::Int)
    Nominal(values::AbstractVector)
"""
struct Nominal{V<:AbstractVector} <: Domain
    values::V
    function Nominal{V}(values::V) where {V<:AbstractVector}
        isempty(values) && throw(ArgumentError("values must be non-empty"))
        return new{V}(values)
    end
end
Nominal(values::AbstractVector) = Nominal{typeof(values)}(values)
Nominal(levels::Int) = Nominal(Base.OneTo(levels))

"""
    Ordinal(levels::Int)
    Ordinal(values::AbstractVector)
"""
struct Ordinal{V<:AbstractVector} <: Domain
    values::V
    function Ordinal{V}(values::V) where {V<:AbstractVector}
        isempty(values) && throw(ArgumentError("values must be non-empty"))
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
"""
struct Continuous{F} <: Domain
    min::Float64
    max::Float64
    transform::F
    function Continuous{F}(lo::Float64, hi::Float64, transform::F) where {F}
        hi > lo || throw(ArgumentError("max ($hi) must be greater than min ($lo)"))
        # Endpoints only: a grid of any size can step over a pole, so sampling the interior
        # wouldn't establish anything it doesn't already.
        (isfinite(transform(lo)) && isfinite(transform(hi))) ||
            throw(ArgumentError("transform must be finite at both ends of [$lo, $hi]; got non-finite values"))
        return new{F}(lo, hi, transform)
    end
end
Continuous(min::Real, max::Real; transform=identity) =
    Continuous{typeof(transform)}(Float64(min), Float64(max), transform)

"""
    length(d::Union{Nominal,Ordinal})
"""
Base.length(d::Union{Nominal,Ordinal}) = length(d.values)

"""
    from_unit(d::Domain, u::Real)
"""
from_unit(d::Union{Nominal,Ordinal}, u::Real) = d.values[clamp(ceil(Int, u * length(d)), 1, length(d))]
from_unit(d::Continuous, u::Real) = d.transform(d.min + clamp(u, 0.0, 1.0) * (d.max - d.min))
