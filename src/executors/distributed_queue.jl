"""
    DistributedQueue(max_concurrency=Sys.CPU_THREADS; spawn_worker)

Runs each trial on its own freshly spawned worker process (via
`spawn_worker()`, which must return the new worker's pid), torn down again
as soon as that one trial resolves -- up to `max_concurrency` trials in
flight at once. No worker is ever reused across trials: this library targets
costly trials (often taking hours), for which process-spawn/precompilation
overhead is negligible, so the stronger isolation of a fresh process per
trial (no risk of one trial's state leaking into another's) is worth more
than the small amortized cost reuse would save.

`spawn_worker` is required, with no default -- there's no generic
implementation that could work for an arbitrary objective. At minimum it
must load this package onto the new worker (a bare `Distributed.addprocs(1)`
leaves a worker with nothing loaded, so even this package's own internals
would fail to resolve there), and in practice it usually also needs to load
whatever packages or helper functions the objective itself depends on --
e.g.:

```julia
spawn_worker = () -> begin
    pid = first(Distributed.addprocs(1))
    Distributed.remotecall_eval(Main, [pid], :(using BigHO; using MyObjectiveDeps))
    return pid
end
```

Tolerating worker death is still the point: if a worker dies mid-trial, that
trial is reported as failing (a `ProcessExitedException` is just another
non-Real outcome to `finalize_entry`, same as any other exception) rather
than the run aborting. `entry.params`/`entry.pre_artefact`/the objective
function itself must all be serializable, since they cross process
boundaries.

A genuine `InterruptException` thrown by the objective itself is still
recognized as one and stops the run gracefully, the same as it would under
`Serial`/`Threaded` -- even though it actually crosses back from the worker
wrapped in a `RemoteException`.
"""
mutable struct DistributedQueue <: AbstractExecutor
    max_concurrency::Int
    spawn_worker::Any
    results::Channel{Tuple{RunEntry,Any}}
    in_flight::Int
    pending_exception::Union{Exception,Nothing}
    tasks::Vector{Task}

    # Defining this inner constructor suppresses Julia's default positional
    # one, which would otherwise let max_concurrency<1 bypass the outer
    # convenience constructor's validation below entirely.
    function DistributedQueue(max_concurrency::Int, spawn_worker, results::Channel{Tuple{RunEntry,Any}}, in_flight::Int, pending_exception::Union{Exception,Nothing}, tasks::Vector{Task})
        max_concurrency >= 1 || throw(ArgumentError("max_concurrency must be >= 1, got $max_concurrency"))
        return new(max_concurrency, spawn_worker, results, in_flight, pending_exception, tasks)
    end
end
function DistributedQueue(max_concurrency::Int=Sys.CPU_THREADS; spawn_worker)
    return DistributedQueue(max_concurrency, spawn_worker, Channel{Tuple{RunEntry,Any}}(Inf), 0, nothing, Task[])
end

function start!(executor::DistributedQueue, ho)
    executor.results = Channel{Tuple{RunEntry,Any}}(Inf)
    executor.in_flight = 0
    executor.pending_exception = nothing
    executor.tasks = Task[]
    return nothing
end

"""
    shutdown!(executor::DistributedQueue)

Waits for every trial this executor ever dispatched to actually resolve --
see [`shutdown!(::Threaded)`](@ref) for why this is necessary at all. There's
no worker pool to tear down here: each trial's own worker is already torn
down by `submit!` as soon as that trial resolves. A genuine InterruptException
(e.g. a second Ctrl+C while already waiting out a hung worker) propagates
rather than being swallowed -- everything else (a task failing for some other
reason) is not this function's concern to report, since any reportable
outcome already went through the results channel.
"""
function shutdown!(executor::DistributedQueue)
    for t in executor.tasks
        try
            wait(t)
        catch e
            e isa InterruptException && rethrow()
        end
    end
    return nothing
end

# Tears down a trial's worker. A non-interrupt failure here (e.g. the worker
# already being unreachable) is logged rather than propagated, so it can't be
# mistaken for -- or silently overwrite -- that trial's own outcome in
# submit! below. A genuine InterruptException still propagates, matching
# every other blocking call in this file (fetch, shutdown!'s wait).
function _teardown_worker(worker)
    try
        Distributed.rmprocs(worker)
    catch e
        e isa InterruptException && rethrow()
        @warn "Failed to tear down worker after trial" exception = e
    end
    return nothing
end

# A genuine InterruptException thrown by the objective itself runs on a
# remote worker, so it crosses back to the driver through fetch() wrapped in
# a RemoteException, not as the raw InterruptException poll()/_take_one!
# check for (confirmed empirically: fetch() on a future whose remote task
# threw InterruptException raises RemoteException(pid, CapturedException(
# InterruptException(), ...)), not InterruptException itself). Unwrapped
# right where it's caught so everything downstream only ever has to
# recognize one shape of "this was really an interrupt" -- any other
# RemoteException (an unrelated remote error, e.g. a deserialization
# failure) is left untouched and still reported as an ordinary failed
# outcome, exactly as before.
_unwrap_remote_interrupt(e) = e isa RemoteException && e.captured.ex isa InterruptException ? e.captured.ex : e

function submit!(executor::DistributedQueue, entry::RunEntry, f)
    executor.in_flight += 1
    t = @async begin
        # outcome is resolved to exactly one value -- the fetched result or
        # the caught exception -- before the single put! below, so a
        # teardown failure in _teardown_worker's finally can never produce a
        # second, phantom result for the same entry (its own non-interrupt
        # failures are absorbed there precisely to prevent that).
        outcome = try
            worker = executor.spawn_worker()
            try
                future = Distributed.remotecall(safe_call, worker, f, entry.params, entry.pre_artefact)
                fetch(future)
            finally
                # Always torn down -- whether this trial succeeded, failed,
                # or the worker itself died underneath it (in which case this
                # is a harmless no-op on an already-gone pid).
                _teardown_worker(worker)
            end
        catch e
            _unwrap_remote_interrupt(e)
        end
        put!(executor.results, (entry, outcome))
    end
    push!(executor.tasks, t)
    return nothing
end

# Takes one result off executor's channel into out, unless it's an
# InterruptException, which is stashed on executor.pending_exception instead
# (see poll below for why). Shared by both call sites in poll so they can't
# drift out of sync with each other.
function _take_one!(executor::DistributedQueue, out)
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
function poll(executor::DistributedQueue)
    # A pending exception from a PREVIOUS call is thrown first -- see
    # poll(::Threaded) for why this deferred-throw scheme is needed at all
    # (a real InterruptException landing inside one of this executor's own
    # supervisory tasks, rather than a worker dying, which is instead
    # forwarded as a normal per-trial failure in _take_one! above).
    executor.pending_exception !== nothing && throw(executor.pending_exception)
    executor.in_flight == 0 && return Tuple{RunEntry,Any}[]
    out = _take_one!(executor, Tuple{RunEntry,Any}[])
    while executor.in_flight > 0 && isready(executor.results)
        _take_one!(executor, out)
    end
    return out
end

capacity(executor::DistributedQueue) = executor.pending_exception === nothing ? executor.max_concurrency - executor.in_flight : 0
