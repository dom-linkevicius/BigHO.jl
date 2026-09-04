"""
    Hyperband(; R, η=3, r_min=1, inner=RandomSampler())

Original Hyperband (Li et al. 2018): one finite pass over `smax=⌊log_η(R/r_min)⌋+1` brackets of decreasing aggressiveness, promoting the top `1/η` of a rung once it's fully told. A [`FixedPlanSampler`](@ref) -- `n` is computed automatically, not resumable via `settarget!`.
Construct via `Hyperoptimizer(objective, candidates, Hyperband(...))`, which prepends reserved `:r` (`Ordinal` over `r_min,...,R`); objective called as `f(r, params...)` (or `f(r, params...; pre_artefact)` if [`Stateful`](@ref)).
"""
const Hyperband = SuccessiveHalving{true}

# Walks brackets from most aggressive (smax+1) down, returning the first actionable decision --
# a fresh draw, a promotion, :wait (blocked, needs a tell!), or :exhausted (nothing left, ever).
# Bracket k has no more decisions to ever make once its top rung is resolved (nothing promotes
# out of the top rung), so once past it, it's never revisited. Pure (no side effects, never calls
# `inner`) so exhausted/blocked/create_run_entry can safely re-derive the same decision the
# callable already resolved for this ask!, without re-triggering a stateful sampler like RandomSampler.
# A rung's promotion gate is "resolved" (every dispatched trial there has a terminal status), not
# "every one succeeded" -- failures just shrink how many advance, capped by min(capacity, told).
function _bracket_decision(s::Hyperband, k::Int, runs)
    R, r_min, η = s.R, s.r_min, s.η
    _dispatched_count(runs, k, 1) < _capacity(R, r_min, η, k, 1) && return (:draw, k)
    for i in 1:(k-1)
        _rung_resolved(R, r_min, η, runs, k, i) || return (:wait,)
        told = _told_sorted(runs, k, i)
        target = min(_capacity(R, r_min, η, k, i + 1), length(told))
        target == 0 && return k > 1 ? _bracket_decision(s, k - 1, runs) : (:exhausted,)
        n_promoted = _dispatched_count(runs, k, i + 1)
        n_promoted < target && return (:promote, k, i, first(told[n_promoted+1]))
    end
    _rung_resolved(R, r_min, η, runs, k, k) || return (:wait,)
    return k > 1 ? _bracket_decision(s, k - 1, runs) : (:exhausted,)
end

# Hyperband's own sampling strategy for a fresh draw: delegate to `inner`, dispatched on its own
# type in sh.jl (e.g. LHSampler needs only fresh rows; RandomSampler doesn't care). A future
# sampler (e.g. BOHB, needing every rung of every bracket to fit its KDE prior) would define its
# own method here instead, not delegating to `inner` at all.
_sample_sh_inner(s::Hyperband, candidates, runs) = _sample_sh_inner(s.inner, candidates, runs)
