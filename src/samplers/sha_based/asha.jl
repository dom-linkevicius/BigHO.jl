# ASHA: a rung is eligible the moment anything's been told at it (never blocks).
_asha_ready(bracket::SHABracket, i::Int) = !isempty(bracket.told[i])

"""
    ASHA(; R, η=3, r_min=1)

Asynchronous Hyperband (Li et al. 2020): same one-pass brackets as [`Hyperband`](@ref), but promotes the instant a trial is verifiably top-`1/η` of what's told so far -- never blocks. Multi-bracket scheduling isn't fully specified by the paper, so this is an interpretation, not a transcription.
Construction, `:r`, and the objective-calling convention are identical to [`Hyperband`](@ref).
"""
const ASHA = SuccessiveHalving{false}

(s::ASHA)(candidates, runs) = _ask(s, _asha_ready, candidates, runs)

exhausted(s::ASHA, ho) = _exhausted(s, _asha_ready)
