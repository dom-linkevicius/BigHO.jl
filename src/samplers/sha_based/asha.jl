# ASHA: a rung is eligible the moment anything's been told at it (never blocks).
_asha_ready(bracket::SHABracket, i::Int) = !isempty(bracket.told[i])

"""
    ASHA(; R, η=3, r_min=1)

Asynchronous Hyperband (Li et al. 2020): the same one-pass bracket sequence as
[`Hyperband`](@ref), but a trial is promoted to the next rung the instant it's verifiably in
the top `1/η` of whatever has been told so far at that rung -- never blocks. Brackets are
independent -- each one's own promotion process keeps running to completion regardless of how
many newer brackets have since started -- and a new bracket becomes active as soon as the
previous one's bottom rung has been fully dispatched (not necessarily told). The paper
doesn't fully specify this multi-bracket scheduling, so this is an interpretation of it, not
a transcription.

Construct via `Hyperoptimizer(objective, candidates, ASHA(...))` -- see [`Hyperband`](@ref)
for the reserved `:r` candidate and objective-calling convention, both identical here.
"""
const ASHA = SuccessiveHalving{false}

(s::ASHA)(candidates, runs) = _ask(s, _asha_ready, candidates, runs)

exhausted(s::ASHA, ho) = _exhausted(s, _asha_ready)
