"""
    ASHA(; R, η=3, r_min=1, inner=RandomSampler())

Asynchronous Hyperband (Li et al. 2020): same one-pass brackets as [`Hyperband`](@ref), but promotes the instant a trial is verifiably top-`1/η` of what's told so far -- never blocks. Multi-bracket scheduling isn't fully specified by the paper, so this is an interpretation, not a transcription.
Construction, `:r`, and the objective-calling convention are identical to [`Hyperband`](@ref).

NOTE: not yet implemented on top of the shared `SuccessiveHalving` redesign -- constructs fine, but has no `_bracket_decision` method yet, so `ask!`/`run!` will `MethodError`.
"""
const ASHA = SuccessiveHalving{false}
