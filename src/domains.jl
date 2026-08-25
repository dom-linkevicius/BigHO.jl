"""
    Domain

Describes *what kind* of variable a hyperparameter is, independent of any
particular sampling algorithm or candidate vector -- continuous, ordinal
(ordered discrete), or categorical (unordered/nominal discrete). This is
metadata, not a sampler: it exists so later machinery (Latin hypercube design,
BOHB's kernel density estimates) knows which kernel/distance to use for a
given dimension without having to reverse-engineer it from a candidate
vector's element type.
"""
abstract type Domain end

"""
    Continuous(min, max)

A continuous real-valued dimension bounded by `[min, max]`.
"""
struct Continuous <: Domain
    min::Float64
    max::Float64
end
Continuous(min::Real, max::Real) = Continuous(Float64(min), Float64(max))

"""
    rand([rng,] d::Continuous) -> Float64

Draw a value uniformly from `[d.min, d.max]`.
"""
Base.rand(rng::Random.AbstractRNG, d::Continuous) = d.min + rand(rng) * (d.max - d.min)
Base.rand(d::Continuous) = rand(Random.default_rng(), d)

"""
    Ordinal(levels)

A discrete dimension with `levels` distinct values that have a meaningful
order (e.g. "low" < "medium" < "high", or an integer count) even though the
values themselves need not be numeric. Order matters for distance-based
methods (Latin hypercube design, ordinal KDE kernels).
"""
struct Ordinal <: Domain
    levels::Int
end

"""
    rand([rng,] d::Ordinal) -> Int

Draw a level index uniformly from `1:d.levels`. `Ordinal` describes the
*kind* of dimension, not the actual candidate values, so a draw is an index
into whatever ordered candidate list this dimension is paired with elsewhere.
"""
Base.rand(rng::Random.AbstractRNG, d::Ordinal) = rand(rng, 1:d.levels)
Base.rand(d::Ordinal) = rand(Random.default_rng(), d)

"""
    Categorical(levels)

A discrete dimension with `levels` distinct nominal values that have no
meaningful order (e.g. a choice of activation function). Any two distinct
levels must be treated as equally "different" by design/KDE methods.
"""
struct Categorical <: Domain
    levels::Int
end

"""
    rand([rng,] d::Categorical) -> Int

Draw a level index uniformly from `1:d.levels`. `Categorical` describes the
*kind* of dimension, not the actual candidate values, so a draw is an index
into whatever nominal candidate list this dimension is paired with elsewhere.
"""
Base.rand(rng::Random.AbstractRNG, d::Categorical) = rand(rng, 1:d.levels)
Base.rand(d::Categorical) = rand(Random.default_rng(), d)
