"""
    LHSampler(; gens=nothing)

Draw all `ho.n` trials at once from an optimized Latin Hypercube design over `ho.candidates`.
A `FixedPlanSampler`: requires `ho.n` to be set, and can't be resumed via `settarget!` -- the design is optimized for one fixed trial count.
Each `Domain` maps directly onto `LatinHypercubeSampling.jl`'s own dimension kinds (`Nominal` -> `Categorical`, `Ordinal`/`Continuous` -> `Continuous`), so no separate `dims=` argument is needed.
A `Continuous`/`Ordinal` domain must have exactly `ho.n` values (the design assigns each exactly once); a `Nominal` domain may have any number of levels.
`weights` on any domain isn't supported.
`gens`, if given, overrides the number of LHC-optimization generations (default: a size-scaled heuristic).
"""
mutable struct LHSampler <: Sampler
    gens::Union{Int,Nothing}
    design::Matrix{Int} # ho.n × ndims, 1-based level indices into each domain's values; empty until init!
end

LHSampler(; gens::Union{Int,Nothing}=nothing) = LHSampler(gens, Matrix{Int}(undef, 0, 0))

function _lhc_dimension(d::Domain)
    d.weights === nothing || throw(ArgumentError("LHSampler doesn't support weighted domains"))
    return d.type === :nominal ? LatinHypercubeSampling.Categorical(length(d.values)) : LatinHypercubeSampling.Continuous()
end

function init!(s::LHSampler, ctx::AskContext)
    ctx.n === nothing &&
        throw(ArgumentError("LHSampler requires ho.n to be set -- it draws a design matrix sized to the total trial count up front"))
    for d in ctx.candidates
        d.type === :nominal || length(d.values) == ctx.n ||
            throw(ArgumentError("LHSampler requires every Continuous/Ordinal domain to have exactly ho.n ($(ctx.n)) values, got $(length(d.values))"))
    end
    dims = [_lhc_dimension(d) for d in ctx.candidates]
    ndims = length(dims)
    gens = something(s.gens, max(1, (1000 * 100 * 2) ÷ ctx.n ÷ ndims))
    initial = LatinHypercubeSampling.randomLHC(ctx.n, dims)
    X, _ = LatinHypercubeSampling.LHCoptim!(initial, gens; dims)
    s.design = X
    return nothing
end

function (s::LHSampler)(ctx::AskContext)
    row = ctx.n_asked + 1
    return [d.values[s.design[row, dim]] for (dim, d) in enumerate(ctx.candidates)]
end

on_tell!(::LHSampler, entry) = nothing
exhausted(s::LHSampler, ho) = length(ho.runs) >= size(s.design, 1)
