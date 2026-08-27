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
