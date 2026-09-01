"""
    Threaded(max_concurrency=Threads.nthreads())

Runs each trial as a separate task via `Threads.@spawn`, up to
`max_concurrency` at once. Warns if only one Julia thread is available --
trials still run concurrently as tasks, but not in true parallel (start
Julia with `--threads=N` for that).
"""
mutable struct Threaded <: AbstractExecutor
    max_concurrency::Int
    results::Channel{Tuple{RunEntry,Any}}
    in_flight::Int
    pending_exception::Union{Exception,Nothing}
    tasks::Vector{Task}

    # Defining this inner constructor suppresses Julia's default positional
    # one, which would otherwise let max_concurrency<1 bypass the outer
    # convenience constructor's validation below entirely (e.g.
    # Threaded(0, Channel{Tuple{RunEntry,Any}}(Inf), 0, nothing, Task[])).
    function Threaded(max_concurrency::Int, results::Channel{Tuple{RunEntry,Any}}, in_flight::Int, pending_exception::Union{Exception,Nothing}, tasks::Vector{Task})
        max_concurrency >= 1 || throw(ArgumentError("max_concurrency must be >= 1, got $max_concurrency"))
        return new(max_concurrency, results, in_flight, pending_exception, tasks)
    end
end
function Threaded(max_concurrency::Int=Threads.nthreads())
    Threads.nthreads() == 1 &&
        @warn "Threaded executor constructed with only 1 Julia thread available; trials will run concurrently as tasks but not in true parallel -- start Julia with `--threads=N` for real concurrency" maxlog = 1
    return Threaded(max_concurrency, Channel{Tuple{RunEntry,Any}}(Inf), 0, nothing, Task[])
end

function start!(executor::Threaded, ho)
    executor.results = Channel{Tuple{RunEntry,Any}}(Inf)
    executor.in_flight = 0
    executor.pending_exception = nothing
    executor.tasks = Task[]
    return nothing
end

"""
    shutdown!(executor::Threaded)

Waits for every trial this executor ever spawned to actually finish -- Julia
has no way to forcibly cancel a running task, so this is the only way to
guarantee nothing is still executing the objective in the background once
`run!` returns (e.g. after an interrupt stopped the run early). A genuine
InterruptException (e.g. a second Ctrl+C while already waiting out a
runaway objective, or one landing inside `submit!`'s own catch-clause
recovery `put!` -- a real yield point -- which makes that trial's task
itself fail and so surfaces here wrapped in a `TaskFailedException`)
propagates rather than being swallowed -- everything else surfacing here is
not this function's concern to report, since every reportable outcome
already went through the results channel: this is best-effort cleanup, not
error reporting.
"""
function shutdown!(executor::Threaded)
    for t in executor.tasks
        try
            wait(t)
        catch e
            if e isa InterruptException
                rethrow()
            elseif e isa TaskFailedException && e.task.result isa InterruptException
                rethrow(e.task.result)
            end
        end
    end
    return nothing
end

function submit!(executor::Threaded, entry::RunEntry, f)
    executor.in_flight += 1
    t = Threads.@spawn begin
        try
            put!(executor.results, (entry, safe_call(f, entry.params, entry.pre_artefact)))
        catch e
            # safe_call only ever lets InterruptException escape uncaught --
            # forward it through the channel rather than letting it just kill
            # this task silently, which would leave in_flight permanently
            # off-by-one and deadlock poll() waiting for a result that could
            # now never arrive.
            put!(executor.results, (entry, e))
        end
    end
    push!(executor.tasks, t)
    return nothing
end

# Takes one result off executor's channel into out, unless it's an
# InterruptException, which is stashed on executor.pending_exception instead
# (see poll below for why). Shared by both call sites in poll so they can't
# drift out of sync with each other.
function _take_one!(executor::Threaded, out)
    entry, outcome = take!(executor.results)
    executor.in_flight -= 1
    if outcome isa InterruptException
        executor.pending_exception = outcome
    else
        push!(out, (entry, outcome))
    end
    return out
end

# Blocks for the first result only if one is actually in flight (per
# AbstractExecutor's `poll` contract); drains any others already sitting in
# the channel so run! doesn't call back in for results that are ready now.
function poll(executor::Threaded)
    # A pending exception from a PREVIOUS call is thrown first, before
    # anything else -- including before the in_flight==0 check below, since
    # the exception can still be pending even after everything else has
    # already been drained and told. Deferring the throw to the *next* call
    # (rather than throwing as soon as it's found) is what lets this call's
    # own already-collected `out` be returned normally instead of discarded:
    # throwing mid-collection would unwind past `out` and lose every
    # successfully-completed result gathered earlier in that same batch.
    executor.pending_exception !== nothing && throw(executor.pending_exception)
    executor.in_flight == 0 && return Tuple{RunEntry,Any}[]
    # Explicitly typed -- a bare `Any[]` (or inferring from the first
    # _take_one!() call) infers its element type from whichever concrete
    # runtime type shows up first (e.g. an ObjectiveOutcome payload), and
    # then errors on `push!` the moment a differently-typed payload (e.g. a
    # caught ErrorException) shows up later in the same batch.
    out = _take_one!(executor, Tuple{RunEntry,Any}[])
    while executor.in_flight > 0 && isready(executor.results)
        _take_one!(executor, out)
    end
    return out
end

capacity(executor::Threaded) = executor.pending_exception === nothing ? executor.max_concurrency - executor.in_flight : 0
