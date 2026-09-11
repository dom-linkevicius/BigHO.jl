"""
    Domain
"""
struct Domain
    type::Symbol
    values::AbstractVector
    weights::Union{Vector{Float64},Nothing}

    function Domain(type::Symbol, values::AbstractVector, weights::Union{Vector{Float64},Nothing})
        type in (:nominal, :ordinal, :continuous_linear, :continuous_arbitrary) ||
            throw(ArgumentError("type must be :nominal, :ordinal, :continuous_linear, or :continuous_arbitrary, got $(repr(type))"))
        isempty(values) && throw(ArgumentError("values must be non-empty"))
        weights === nothing || length(weights) == length(values) ||
            throw(ArgumentError("weights must have length $(length(values)) (one per level), got $(length(weights))"))
        weights === nothing || (all(>=(0), weights) && any(>(0), weights)) ||
            throw(ArgumentError("weights must be non-negative, with at least one strictly positive; got $(count(<(0), weights)) negative and $(count(==(0), weights)) zero, out of $(length(weights))"))
        return new(type, values, weights)
    end
end

"""
    DEFAULT_DOMAIN_RNG
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
"""
Base.rand(rng::Random.AbstractRNG, d::Domain) = _draw(rng, d, d.weights)

"""
    x in d::Domain
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
"""
Nominal(levels::Int; weights::Union{Vector{Float64},Nothing}=nothing) = _from_levelcount(:nominal, levels, weights)
Nominal(values::AbstractVector; weights::Union{Vector{Float64},Nothing}=nothing) = _from_values(:nominal, values, weights)

"""
    Ordinal(levels::Int; weights=nothing)
    Ordinal(values::AbstractVector; weights=nothing)
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
"""
function Continuous(min::Real, max::Real, dt::Real; weights::Union{Vector{Float64},Nothing}=nothing)
    min, max, dt = Float64(min), Float64(max), Float64(dt)
    max >= min || throw(ArgumentError("max ($max) must be >= min ($min)"))
    dt > 0 || throw(ArgumentError("dt must be > 0, got $dt"))
    return _continuous_domain(:continuous_linear, min:dt:max, weights)
end
function Continuous(values::AbstractVector{<:Real}; weights::Union{Vector{Float64},Nothing}=nothing)
    issorted(values) && allunique(values) ||
        throw(ArgumentError("Continuous(values) requires strictly increasing values (no duplicates); got $values"))
    return _continuous_domain(:continuous_arbitrary, values, weights)
end
function _continuous_domain(type::Symbol, values::AbstractVector{<:Real}, weights)
    length(values) < 5 && @warn "Continuous domain has fewer than 5 candidate values; consider using Ordinal instead"
    return Domain(type, Float64.(values), weights)
end
