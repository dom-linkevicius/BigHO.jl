"""
    AbstractExecutor

Pluggable evaluation backend: runs the objective somewhere (serially, on a
thread, on a Distributed worker, ...) and hands outcomes back via `poll`.
"""
abstract type AbstractExecutor end

start!(::AbstractExecutor, ho) = nothing
shutdown!(::AbstractExecutor) = nothing

"""
    safe_call(f, params, pre_artefact) -> value_or_exception

Calls the objective, catching any thrown exception and returning it instead.
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

Runs each trial synchronously, one at a time.
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
