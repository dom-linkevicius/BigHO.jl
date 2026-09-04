# Hyperband: only once every trial dispatched at that rung has been told a result.
_hyperband_ready(bracket::SHABracket, i::Int) = length(bracket.told[i]) >= bracket.capacities[i]

"""
    Hyperband(; R, η=3, r_min=1)

Original Hyperband (Li et al. 2018): one finite pass over `smax=⌊log_η(R/r_min)⌋+1` brackets of decreasing aggressiveness, promoting the top `1/η` of a rung once it's fully told. A [`FixedPlanSampler`](@ref) -- `n` is computed automatically, not resumable via `settarget!`.
Construct via `Hyperoptimizer(objective, candidates, Hyperband(...))`, which prepends reserved `:r` (`Ordinal` over `r_min,...,R`); objective called as `f(r, params...)` (or `f(r, params...; pre_artefact)` if [`Stateful`](@ref)).
"""
const Hyperband = SuccessiveHalving{true}

(s::Hyperband)(candidates, runs) = _ask(s, _hyperband_ready, candidates, runs)

exhausted(s::Hyperband, ho) = _exhausted(s, _hyperband_ready)
