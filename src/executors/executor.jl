"""
    AbstractExecutor

Pluggable evaluation backend: runs the objective somewhere (serially, on a
thread, on a Distributed worker, ...) and hands outcomes back via `poll`.
Concrete executors must implement `start!`, `shutdown!`, `submit!`, `poll`,
and `capacity` -- none of these have a default, so a missing one is a loud
`MethodError` rather than a silent no-op.
"""
abstract type AbstractExecutor end

"""
    start!(executor, ho)

Called once before `run!`'s main loop begins.
"""
function start! end

"""
    shutdown!(executor)

Called once after `run!`'s main loop ends, successfully or not.
"""
function shutdown! end

"""
    submit!(executor, entry::RunEntry, f)

Dispatch `entry` for evaluation via objective `f`, however this executor
runs objectives.
"""
function submit! end

"""
    poll(executor) -> Vector{Tuple{RunEntry,Any}}

Collect completed `(entry, outcome)` pairs. `run!` calls this in a tight
loop with no yielding of its own, so `poll` must block until at least one
result is ready whenever something is still in flight (like `Base.take!` on
a `Channel`), rather than returning an empty list -- otherwise `run!`
busy-spins instead of idling while work completes.
"""
function poll end

"""
    capacity(executor) -> Int

How many more trials this executor can accept right now.
"""
function capacity end

"""
    safe_call(f, params, pre_artefact) -> value_or_exception

Calls the objective, catching any thrown exception and returning it instead
-- except `InterruptException`, which propagates so `run!` can stop
gracefully instead of recording the interruption as a failed trial.
"""
function safe_call(f, params, pre_artefact)
    try
        call_objective(f, params, pre_artefact)
    catch e
        e isa InterruptException && rethrow()
        e
    end
end
