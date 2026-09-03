"""
    LHSampler(; gens=nothing)

Draw all `ho.n` trials at once from an optimized Latin Hypercube design over `ho.candidates`.
A `FixedPlanSampler`: can't be resumed via `settarget!` -- the design is optimized for one fixed trial count.
Each `Domain` maps directly onto `LatinHypercubeSampling.jl`'s own dimension kinds (`Nominal`/`Ordinal` -> `Categorical`, `Continuous(min,max,dt)` -> `Continuous`), so no separate `dims=` argument is needed. `Continuous(values)` (arbitrary spacing) isn't supported.
Construct via `Hyperoptimizer(objective, candidates, LHSampler(); n=...)`, which rebuilds `Continuous(min,max,dt)` domains to match `n` -- see that constructor's own docstring.
`weights` on any domain isn't supported.
`gens`, if given, overrides the number of LHC-optimization generations directly; otherwise it defaults to `10` times the product of each domain's own effective level count (`n` for `Continuous`, its actual level count for `Nominal`/`Ordinal`).
Its optimization history is inspectable via [`get_lhs_optim_history`](@ref).
"""
struct LHSampler <: Sampler
    gens::Union{Int,Nothing}
    design::Matrix{Int} # ho.n × ndims, 1-based level indices into each domain's values; empty until init
    history::Vector{Float64} # per-generation best Audze-Eglais fitness from LHCoptim!; empty until init
end

LHSampler(; gens::Union{Int,Nothing}=nothing) = LHSampler(gens, Matrix{Int}(undef, 0, 0), Float64[])

# A dimension's own contribution to the design's permutation search space -- a Continuous
# column is a full n-permutation (n!), while a Categorical(levels) column's distinct
# arrangements are a much smaller multinomial coefficient, closer to its own level count.
_effective_levels(d::Domain, n::Int) = d.type === :continuous_linear ? n : length(d.values)

function _lhc_dimension(d::Domain)
    d.weights === nothing || throw(ArgumentError("LHSampler doesn't support weighted domains"))
    return d.type === :continuous_linear ? LatinHypercubeSampling.Continuous() : LatinHypercubeSampling.Categorical(length(d.values))
end

# Only a :continuous_linear domain gets rebuilt (it has a well-defined range to rebuild over);
# :continuous_arbitrary is left as-is here and rejected below in init's validation instead.
function _linearize_for_lhs(candidates::NamedTuple, n::Int)
    names = keys(candidates)
    vals = map(names, values(candidates)) do name, d
        if d.type === :continuous_linear && length(d.values) != n
            lo, hi = extrema(d.values)
            @warn "LHSampler: overriding `$name`'s $(length(d.values))-value grid with $n linearly-spaced values over [$lo, $hi]"
            # Built directly (not via the public Continuous(values) constructor) so the
            # rebuilt domain keeps the :continuous_linear tag instead of becoming :continuous_arbitrary.
            _continuous_domain(:continuous_linear, collect(Float64, LinRange(lo, hi, n)), nothing)
        else
            d
        end
    end
    return NamedTuple{names}(vals)
end

_discrete_product(candidates) = prod((length(d.values) for d in candidates if d.type in (:nominal, :ordinal)); init=1)

function init(s::LHSampler, ctx::AskContext)
    for d in ctx.candidates
        d.type !== :continuous_arbitrary ||
            throw(ArgumentError("LHSampler doesn't support Continuous(values) domains (arbitrary spacing) -- use Continuous(min, max, dt), or apply the nonlinear transform inside the objective itself"))
        d.type !== :continuous_linear || length(d.values) == ctx.n ||
            throw(ArgumentError("LHSampler requires every Continuous domain to have exactly n ($(ctx.n)) values, got $(length(d.values))"))
    end
    product = _discrete_product(ctx.candidates)
    ctx.n < product && @warn "LHSampler: n ($(ctx.n)) is less than the number of discrete-variable combinations ($product) -- not every combination can be covered with this budget"
    dims = [_lhc_dimension(d) for d in ctx.candidates]
    gens = something(s.gens, 10 * prod(_effective_levels(d, ctx.n) for d in ctx.candidates))
    initial = LatinHypercubeSampling.randomLHC(ctx.n, dims)
    @info "LHC optimization via a genetic algorithm with $gens generations is starting, may take a few minutes. You can inspect the point spread optimization results for convergence using get_lhs_optim_history(ho)"
    X, hist = LatinHypercubeSampling.LHCoptim!(initial, gens; dims)
    _warn_not_converged(s, hist)
    ctx.n >= product && _warn_missing_combinations(X, ctx.candidates)
    return LHSampler(s.gens, X, hist)
end

# The best design found so late that more generations may well have kept improving it.
function _warn_not_converged(s::LHSampler, hist)
    best_gen = findfirst(==(hist[end]), hist)
    if best_gen > 0.9 * length(hist)
        suggestion = s.gens === nothing ?
            "passing gens explicitly instead of the computed default ($(length(hist) - 1))" :
            "increasing gens (currently $(s.gens))"
        @warn "LHSampler: the optimization may not have converged (best design found in the last 10% of $(length(hist)) generations) -- consider $suggestion"
    end
    return nothing
end

# Only meaningful when full coverage is theoretically achievable (ctx.n >= product) -- checks
# whether the GA-optimized design actually achieved it, since that isn't guaranteed.
function _warn_missing_combinations(design::Matrix{Int}, candidates)
    discrete_dims = findall(d -> d.type in (:nominal, :ordinal), candidates)
    isempty(discrete_dims) && return nothing
    covered = Set(Tuple(row[discrete_dims]) for row in eachrow(design))
    all_combos = vec(collect(Iterators.product((1:length(candidates[dim].values) for dim in discrete_dims)...)))
    missing_combos = filter(c -> c ∉ covered, all_combos)
    isempty(missing_combos) && return nothing
    shown = [Tuple(candidates[dim].values[c[i]] for (i, dim) in enumerate(discrete_dims)) for c in first(missing_combos, 20)]
    suffix = length(missing_combos) > 20 ? " (and $(length(missing_combos) - 20) more)" : ""
    @warn "LHSampler: the optimized design doesn't cover every discrete-variable combination$suffix" missing = shown
    return nothing
end

function (s::LHSampler)(ctx::AskContext)
    row = ctx.n_asked + 1
    return [d.values[s.design[row, dim]] for (dim, d) in enumerate(ctx.candidates)]
end

on_tell!(::LHSampler, entry) = nothing
exhausted(s::LHSampler, ho) = length(ho.runs) >= size(s.design, 1)

"""
    get_lhs_optim_history(ho) -> Vector{Float64}

The per-generation best Audze-Eglais fitness from `LHCoptim!`, for inspecting whether the optimization converged. Only defined for a Hyperoptimizer using [`LHSampler`](@ref).
"""
function get_lhs_optim_history(ho)
    ho.sampler isa LHSampler ||
        throw(ArgumentError("get_lhs_optim_history: ho.sampler is a $(typeof(ho.sampler)), not LHSampler"))
    return ho.sampler.history
end
