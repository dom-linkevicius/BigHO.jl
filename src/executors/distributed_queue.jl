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

If `spawn_worker` raises *after* it has already provisioned a worker process
(e.g. `addprocs` itself succeeded but a later setup step, like the
`remotecall_eval` above, failed), it is responsible for tearing that worker
down itself before the exception escapes -- `submit!` is never given a pid it
wasn't returned, so it has no way to reap it. Such an exception is otherwise
treated exactly like any other per-trial failure: the trial is recorded as
`Failed` and the run continues.

A genuine `InterruptException` thrown by the objective itself is still
recognized as one and stops the run gracefully, the same as it would under
`Serial`/`Threaded` -- even though it actually crosses back from the worker
wrapped in a `RemoteException`.
"""
mutable struct DistributedQueue <: AbstractExecutor
    max_concurrency::Int
    spawn_worker::Any
    teardown_timeout::Real
    results::Channel{Tuple{RunEntry,Any}}
    in_flight::Int
    pending_exception::Union{Exception,Nothing}
    tasks::Vector{Task}

    # Defining this inner constructor suppresses Julia's default positional
    # one, which would otherwise let max_concurrency<1 bypass the outer
    # convenience constructor's validation below entirely.
    function DistributedQueue(max_concurrency::Int, spawn_worker, teardown_timeout::Real, results::Channel{Tuple{RunEntry,Any}}, in_flight::Int, pending_exception::Union{Exception,Nothing}, tasks::Vector{Task})
        max_concurrency >= 1 || throw(ArgumentError("max_concurrency must be >= 1, got $max_concurrency"))
        teardown_timeout > 0 ||
            throw(ArgumentError("teardown_timeout must be a positive number of seconds (got $teardown_timeout) -- 0 disables Distributed.rmprocs's blocking wait entirely, silently reinstating an unbounded worker_lock hold"))
        return new(max_concurrency, spawn_worker, teardown_timeout, results, in_flight, pending_exception, tasks)
    end
end
"""
    DistributedQueue(max_concurrency=Sys.CPU_THREADS; spawn_worker, teardown_timeout=30)

`teardown_timeout` bounds (in seconds) how long a single trial's teardown will
wait for its worker to actually terminate -- see `_teardown_worker` for why
this matters. The default roughly matches `LocalManager`'s own kill()
escalation window; a custom `ClusterManager` whose job cancellation takes
longer should raise it. Must be a positive, non-NaN number: `0` is rejected
rather than accepted as "don't wait" -- to `Distributed.rmprocs`, `waitfor=0`
isn't a very short timeout, it's a *different* code path (an unawaited,
fire-and-forget background removal with no bound at all), which would
silently reinstate the exact unbounded hang this field exists to prevent.
`Inf` is allowed as a deliberate, informed way to opt back into an unbounded
wait, since it takes the normal bounded-wait code path with an effectively
infinite bound, not the `waitfor==0` special case.
"""
function DistributedQueue(max_concurrency::Int=Sys.CPU_THREADS; spawn_worker, teardown_timeout::Real=30)
    return DistributedQueue(max_concurrency, spawn_worker, teardown_timeout, Channel{Tuple{RunEntry,Any}}(Inf), 0, nothing, Task[])
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
down by `submit!` once that trial's outcome has already been reported. A
genuine InterruptException (e.g. a second Ctrl+C while already waiting out a
hung worker, or one landing during a trial's own post-outcome teardown, which
makes that trial's task itself fail and so surfaces here wrapped in a
`TaskFailedException`) propagates rather than being swallowed -- everything
else (a task failing for some other reason) is not this function's concern to
report, since any reportable outcome already went through the results
channel.
"""
function shutdown!(executor::DistributedQueue)
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

# Tears down a trial's worker, bounded by timeout (rmprocs's own waitfor) so
# one worker that never cleanly acknowledges termination can't hold Julia's
# process-global Distributed.worker_lock forever -- which would otherwise
# also block every OTHER trial's addprocs/rmprocs calls, since that lock is
# shared process-wide, not per-worker (confirmed against the Distributed
# stdlib's own source). A non-interrupt failure here (e.g. the worker already
# being unreachable, or the timeout itself) is logged rather than propagated,
# so it can't be mistaken for -- or silently overwrite -- that trial's own
# outcome, which by the time this runs (see submit! below) has already been
# reported regardless. A genuine InterruptException still propagates,
# matching every other blocking call in this file (fetch, shutdown!'s wait).
function _teardown_worker(worker, timeout)
    try
        Distributed.rmprocs(worker; waitfor=timeout)
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
        # the caught exception -- and put! onto the results channel BEFORE
        # teardown runs below, so a genuine interrupt landing during teardown
        # can never retroactively override an already-obtained outcome (Julia
        # finally-block semantics would otherwise let it do exactly that).
        worker = nothing
        outcome = try
            worker = executor.spawn_worker()
            future = Distributed.remotecall(safe_call, worker, f, entry.params, entry.pre_artefact)
            fetch(future)
        catch e
            _unwrap_remote_interrupt(e)
        end
        put!(executor.results, (entry, outcome))
        # Always torn down -- whether this trial succeeded, failed, or the
        # worker itself died underneath it (in which case this is a harmless
        # no-op on an already-gone pid) -- unless spawn_worker() itself never
        # returned a pid (see the module docstring's note on spawn_worker's
        # own cleanup responsibility in that case). A genuine InterruptException
        # here escapes this task uncaught; shutdown! recognizes and rethrows it.
        worker !== nothing && _teardown_worker(worker, executor.teardown_timeout)
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
