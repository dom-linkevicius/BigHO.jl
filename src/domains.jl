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
    Continuous(min, max; log=false)

A continuous real-valued dimension bounded by `[min, max]`. `log` marks the
dimension as naturally log-uniform (e.g. a learning rate spanning several
orders of magnitude), which downstream samplers/KDEs can use to sample or
estimate density in log-space instead of linear space.
"""
struct Continuous <: Domain
    min::Float64
    max::Float64
    log::Bool
end
Continuous(min::Real, max::Real; log::Bool=false) = Continuous(Float64(min), Float64(max), log)

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
    Categorical(levels)

A discrete dimension with `levels` distinct nominal values that have no
meaningful order (e.g. a choice of activation function). Any two distinct
levels must be treated as equally "different" by design/KDE methods.
"""
struct Categorical <: Domain
    levels::Int
end
