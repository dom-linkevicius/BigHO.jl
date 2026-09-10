"""
    LHSampler(; gens)

Draw all `ho.n` trials at once from an optimized Latin Hypercube design over `ho.candidates`.
A `FixedPlanSampler`: can't be resumed via `settarget!` -- the design is optimized for one fixed trial count.
Each `Domain` maps directly onto `LatinHypercubeSampling.jl`'s own dimension kinds (`Nominal`/`Ordinal` -> `Categorical`, `Continuous` -> `Continuous`), so no separate `dims=` argument is needed.
Construct via `Hyperoptimizer(objective, candidates, LHSampler(gens=...); n=...)`.
`gens` (the number of LHC-optimization generations) must be passed explicitly -- there's no default, since a reasonable value depends heavily on `n` and dimensionality; use [`get_lhs_optim_history`](@ref) to check whether it converged.
"""
struct LHSampler <: Sampler
    gens::Int
    design::Matrix{Int} # ho.n × ndims, 1-based level indices into each domain's values; empty until init
    history::Vector{Float64} # per-generation best Audze-Eglais fitness from LHCoptim!; empty until init
end

LHSampler(; gens::Int) = LHSampler(gens, Matrix{Int}(undef, 0, 0), Float64[])

_lhc_dimension(::Continuous) = LatinHypercubeSampling.Continuous()
_lhc_dimension(d::Union{Nominal,Ordinal}) = LatinHypercubeSampling.Categorical(length(d))

_discrete_product(candidates) = prod((length(d) for d in candidates if d isa Union{Nominal,Ordinal}); init=1)

function init(s::LHSampler, candidates, n)
    product = _discrete_product(candidates)
    n < product && @warn "LHSampler: n ($n) is less than the number of discrete-variable combinations ($product) -- not every combination can be covered with this budget"
    dims = [_lhc_dimension(d) for d in candidates]
    initial = LatinHypercubeSampling.randomLHC(n, dims)
    @info "LHC optimization via a genetic algorithm with $(s.gens) generations is starting, may take a few minutes. You can inspect the point spread optimization results for convergence using get_lhs_optim_history(ho)"
    X, hist = LatinHypercubeSampling.LHCoptim!(initial, s.gens; dims)
    n >= product && _warn_missing_combinations(X, candidates)
    return LHSampler(s.gens, X, hist)
end

# Only meaningful when full coverage is theoretically achievable (n >= product) -- checks
# whether the GA-optimized design actually achieved it, since that isn't guaranteed.
function _warn_missing_combinations(design::Matrix{Int}, candidates)
    discrete_dims = findall(d -> d isa Union{Nominal,Ordinal}, candidates)
    isempty(discrete_dims) && return nothing
    covered = Set(Tuple(row[discrete_dims]) for row in eachrow(design))
    all_combos = vec(collect(Iterators.product((1:length(candidates[dim]) for dim in discrete_dims)...)))
    missing_combos = filter(c -> c ∉ covered, all_combos)
    isempty(missing_combos) && return nothing
    shown = [Tuple(candidates[dim].values[c[i]] for (i, dim) in enumerate(discrete_dims)) for c in first(missing_combos, 20)]
    suffix = length(missing_combos) > 20 ? " (and $(length(missing_combos) - 20) more)" : ""
    @warn "LHSampler: the optimized design doesn't cover every discrete-variable combination$suffix" missing = shown
    return nothing
end

function (s::LHSampler)(candidates, runs)
    row = length(runs) + 1
    n = size(s.design, 1)
    return [_lhc_unit(d, s.design[row, dim], n) for (dim, d) in enumerate(candidates)]
end

# A design column holds 1-based stratum indices: over the domain's own levels for a discrete
# dimension, over all n rows for a continuous one (LatinHypercubeSampling.Continuous() permutes
# 1:n). Either way the proposal is the stratum's centre in [0,1].
_lhc_unit(d::Union{Nominal,Ordinal}, idx::Int, ::Int) = (idx - 0.5) / length(d)
_lhc_unit(::Continuous, idx::Int, n::Int) = (idx - 0.5) / n

on_tell!(::LHSampler, runs, entry) = nothing
exhausted(s::LHSampler, ho) = length(ho.runs) >= size(s.design, 1)
blocked(::LHSampler, ho) = false
create_run_entry(::LHSampler, ho, id, params, unit_params) = RunEntry(id, params, unit_params)

"""
    get_lhs_optim_history(ho) -> Vector{Float64}

The per-generation best Audze-Eglais fitness from `LHCoptim!`, for inspecting whether the optimization converged. Only defined for a Hyperoptimizer using [`LHSampler`](@ref).
"""
function get_lhs_optim_history(ho)
    ho.sampler isa LHSampler ||
        throw(ArgumentError("get_lhs_optim_history: ho.sampler is a $(typeof(ho.sampler)), not LHSampler"))
    return ho.sampler.history
end
