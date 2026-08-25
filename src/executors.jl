"""
    AbstractExecutor

Pluggable evaluation backend. An executor's only job is to run
`objective(trial.params...)` somewhere (serially, on a thread, on a Distributed
worker, ...) and hand the outcome back via `poll` — it never touches `ho`.
Swapping executors requires no change to samplers or user objectives.
"""
abstract type AbstractExecutor end

start!(::AbstractExecutor, ho) = nothing
shutdown!(::AbstractExecutor) = nothing

"""
    safe_call(f, params) -> value_or_exception

Uniform boundary that turns a thrown exception into an ordinary return value,
so every executor's `poll` can guarantee it never throws.
"""
function safe_call(f, params)
    try
        f(params...)
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
    buffer::Vector{Tuple{Trial,Any}}
end
Serial() = Serial(Tuple{Trial,Any}[])

function submit!(executor::Serial, trial::Trial, f)
    push!(executor.buffer, (trial, safe_call(f, trial.params)))
    return nothing
end

function poll(executor::Serial)
    out = copy(executor.buffer)
    empty!(executor.buffer)
    return out
end

capacity(executor::Serial) = isempty(executor.buffer) ? 1 : 0
