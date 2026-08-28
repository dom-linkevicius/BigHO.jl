"""
    DistributedQueue(max_concurrency=Sys.CPU_THREADS; spawn_worker=() -> first(Distributed.addprocs(1)))

Runs each trial on its own freshly spawned worker process (via
`spawn_worker()`, which must return the new worker's pid), torn down again
as soon as that one trial resolves -- up to `max_concurrency` trials in
flight at once. No worker is ever reused across trials: this library targets
costly trials (often taking hours), for which process-spawn/precompilation
overhead is negligible, so the stronger isolation of a fresh process per
trial (no risk of one trial's state leaking into another's) is worth more
than the small amortized cost reuse would save.

Tolerating worker death is still the point: if a worker dies mid-trial, that
trial is reported as failing (a `ProcessExitedException` is just another
non-Real outcome to `finalize_entry`, same as any other exception) rather
than the run aborting. `entry.params`/`entry.pre_artefact` must be
serializable, since they cross process boundaries.
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
function DistributedQueue(max_concurrency::Int=Sys.CPU_THREADS; spawn_worker=() -> first(Distributed.addprocs(1)))
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
down by `submit!` as soon as that trial resolves.
"""
function shutdown!(executor::DistributedQueue)
    for t in executor.tasks
        try
            wait(t)
        catch
        end
    end
    return nothing
end

function submit!(executor::DistributedQueue, entry::RunEntry, f)
    executor.in_flight += 1
    t = @async begin
        try
            worker = executor.spawn_worker()
            try
                future = Distributed.remotecall(safe_call, worker, f, entry.params, entry.pre_artefact)
                put!(executor.results, (entry, fetch(future)))
            finally
                # Always torn down -- whether this trial succeeded, failed,
                # or the worker itself died underneath it (in which case this
                # is a harmless no-op on an already-gone pid).
                Distributed.rmprocs(worker)
            end
        catch e
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
