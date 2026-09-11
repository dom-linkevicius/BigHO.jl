"""
    AbstractExecutor
"""
abstract type AbstractExecutor end

"""
    start!(executor, ho)
"""
function start! end

"""
    shutdown!(executor)
"""
function shutdown! end

"""
    submit!(executor, entry::RunEntry, f)
"""
function submit! end

"""
    poll(executor) -> Vector{Tuple{RunEntry,Any}}
"""
function poll end

"""
    capacity(executor) -> Int
"""
function capacity end

"""
    safe_call(f, params, pre_artefact) -> value_or_exception
"""
function safe_call(f, params, pre_artefact)
    try
        call_objective(f, params, pre_artefact)
    catch e
        e isa InterruptException && rethrow()
        e
    end
end
