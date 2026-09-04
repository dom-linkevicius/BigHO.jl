# Hyperband: only once every trial dispatched at that rung has been told a result.
_hyperband_ready(bracket::SHABracket, i::Int) = length(bracket.told[i]) >= bracket.capacities[i]

"""
    Hyperband(; R, η=3, r_min=1)

The original Hyperband (Li et al. 2018): a single finite pass through brackets of
decreasing aggressiveness (most-to-least), promoting the top `1/η` of a rung to the next
only once every trial dispatched at that rung has been told a result. `smax = ⌊log_η(R/r_min)⌋`
brackets are each run exactly once -- the total trial count is fully determined by
`R`/`η`/`r_min` alone (see [`Hyperoptimizer`](@ref)'s dedicated constructor, which computes
it automatically; a [`FixedPlanSampler`](@ref), not resumable via `settarget!`).

Construct via `Hyperoptimizer(objective, candidates, Hyperband(...))`, which prepends a
reserved `:r` candidate (an `Ordinal` over the resource levels `r_min, r_min*η, ..., R`) --
the objective is called as `f(r, params...)` (or, if [`Stateful`](@ref), `f(r, params...; pre_artefact)`,
receiving its own prior `post_artefact` when resumed after a promotion).
"""
const Hyperband = SuccessiveHalving{true}

(s::Hyperband)(candidates, runs) = _ask(s, _hyperband_ready, candidates, runs)

exhausted(s::Hyperband, ho) = _exhausted(s, _hyperband_ready)
