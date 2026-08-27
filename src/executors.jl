"""
    AbstractExecutor

Pluggable evaluation backend. An executor's only job is to run the
objective somewhere (serially, on a thread, on a Distributed worker, ...) --
via [`call_objective`](@ref), so `pre_artefact` threading works uniformly --
and hand the outcome back via `poll` — it never touches `ho`. Swapping
executors requires no change to samplers or user objectives.
"""
abstract type AbstractExecutor end

start!(::AbstractExecutor, ho) = nothing
shutdown!(::AbstractExecutor) = nothing

"""
    safe_call(f, params, pre_artefact) -> value_or_exception

Uniform boundary that turns a thrown exception into an ordinary return value,
so every executor's `poll` can guarantee it never throws.
"""
function safe_call(f, params, pre_artefact)
    try
        call_objective(f, params, pre_artefact)
    catch e
        e
    end
end

"""
    Serial()

Runs each trial immediately, synchronously, one at a time — reproduces plain
sequential ask/tell semantics. `capacity` forces strict alternation: only one
trial may be in flight before it must be polled.
"""
mutable struct Serial <: AbstractExecutor
    buffer::Vector{Tuple{RunEntry,Any}}
end
Serial() = Serial(Tuple{RunEntry,Any}[])

function submit!(executor::Serial, entry::RunEntry, f)
    push!(executor.buffer, (entry, safe_call(f, entry.params, entry.pre_artefact)))
    return nothing
end

function poll(executor::Serial)
    out = copy(executor.buffer)
    empty!(executor.buffer)
    return out
end

capacity(executor::Serial) = isempty(executor.buffer) ? 1 : 0
