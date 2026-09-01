"""
    DistributedQueue(max_concurrency=Sys.CPU_THREADS; spawn_worker, setup_worker=Returns(nothing), teardown_timeout=30)

Runs each trial on its own freshly spawned worker process, torn down again as soon as that trial resolves, up to `max_concurrency` trials in flight at once. No worker is ever reused across trials.

`spawn_worker()` must return the new worker's pid; it's called synchronously in `submit!`, before that trial's task even exists, so the pid is always known immediately. `setup_worker(pid)` then runs asynchronously to prepare the worker (load packages, etc.) -- defaults to a no-op. e.g.:

```julia
spawn_worker = () -> first(Distributed.addprocs(1))
setup_worker = pid -> Distributed.remotecall_eval(Main, [pid], :(using BigHO; using MyObjectiveDeps))
```

A worker dying mid-trial, or `spawn_worker`/`setup_worker` failing, is reported as an ordinary `Failed` trial rather than aborting the run. `entry.params`/`entry.pre_artefact`/the objective itself must all be serializable.

A local `InterruptException` (e.g. Ctrl+C) aborts the run like any other executor; one raised by the objective itself, running remotely, can't reach the driver and is just an ordinary `Failed` trial.
"""
mutable struct DistributedQueue <: AbstractExecutor
    max_concurrency::Int
    spawn_worker::Any
    setup_worker::Any
    teardown_timeout::Real
    results::Channel{Tuple{RunEntry,Any}}
    in_flight::Int
    pending_exception::Union{Exception,Nothing}
    # Each trial's pid alongside its task, so shutdown! can kill the worker directly.
    tasks::Vector{Tuple{Int,Task}}

    # Suppresses the default positional constructor, which would bypass validation below.
    function DistributedQueue(max_concurrency::Int, spawn_worker, setup_worker, teardown_timeout::Real, results::Channel{Tuple{RunEntry,Any}}, in_flight::Int, pending_exception::Union{Exception,Nothing}, tasks::Vector{Tuple{Int,Task}})
        max_concurrency >= 1 || throw(ArgumentError("max_concurrency must be >= 1, got $max_concurrency"))
        teardown_timeout > 0 ||
            throw(ArgumentError("teardown_timeout must be a positive number of seconds (got $teardown_timeout) -- 0 disables Distributed.rmprocs's blocking wait entirely, silently reinstating an unbounded worker_lock hold"))
        return new(max_concurrency, spawn_worker, setup_worker, teardown_timeout, results, in_flight, pending_exception, tasks)
    end
end
"""
    DistributedQueue(max_concurrency=Sys.CPU_THREADS; spawn_worker, setup_worker=Returns(nothing), teardown_timeout=30)

`teardown_timeout` bounds (in seconds) how long `submit!`'s `rmprocs` call waits for a worker to terminate. Must be positive: `0` means something different to `rmprocs` (unbounded fire-and-forget removal), not "don't wait", so it's rejected -- use `Inf` for that instead.
"""
function DistributedQueue(max_concurrency::Int=Sys.CPU_THREADS; spawn_worker, setup_worker=Returns(nothing), teardown_timeout::Real=30)
    return DistributedQueue(max_concurrency, spawn_worker, setup_worker, teardown_timeout, Channel{Tuple{RunEntry,Any}}(Inf), 0, nothing, Tuple{Int,Task}[])
end

function start!(executor::DistributedQueue, ho)
    executor.results = Channel{Tuple{RunEntry,Any}}(Inf)
    executor.in_flight = 0
    executor.pending_exception = nothing
    executor.tasks = Tuple{Int,Task}[]
    return nothing
end

"""
    shutdown!(executor::DistributedQueue)

Shuts down NOW: any trial not yet done has its worker killed directly rather than waited for. See [`shutdown!(::Threaded)`](@ref) for why every task is still waited on afterward, and why a failed task's exception propagates rather than being swallowed.
"""
function shutdown!(executor::DistributedQueue)
    for (pid, t) in executor.tasks
        if !istaskdone(t)
            try
                Distributed.rmprocs(pid)
            catch
            end
        end
    end
    for (_, t) in executor.tasks
        wait(t)
    end
    return nothing
end

function submit!(executor::DistributedQueue, entry::RunEntry, f)
    executor.in_flight += 1
    # Synchronous, so the pid (or failure) is known before any task exists.
    worker = try
        executor.spawn_worker()
    catch e
        e
    end
    if worker isa Exception
        # An ordinary Failed trial; no worker/task exists, so report it directly.
        put!(executor.results, (entry, worker))
        return nothing
    end
    t = @async begin
        # Resolved and put! BEFORE teardown, so a later interrupt can't override it.
        outcome = try
            executor.setup_worker(worker)
            future = Distributed.remotecall(safe_call, worker, f, entry.params, entry.pre_artefact)
            fetch(future)
        catch e
            e
        end
        put!(executor.results, (entry, outcome))
        # Always torn down, even on failure -- a no-op if the worker's already gone.
        Distributed.rmprocs(worker; waitfor=executor.teardown_timeout)
    end
    push!(executor.tasks, (worker, t))
    return nothing
end

# Takes one result into out, unless it's an InterruptException (stashed on pending_exception instead).
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

# Blocks for the first result only if something's in flight; drains any others already ready.
function poll(executor::DistributedQueue)
    # A pending exception from a PREVIOUS call is thrown first -- see poll(::Threaded).
    executor.pending_exception !== nothing && throw(executor.pending_exception)
    executor.in_flight == 0 && return Tuple{RunEntry,Any}[]
    out = _take_one!(executor, Tuple{RunEntry,Any}[])
    while executor.in_flight > 0 && isready(executor.results)
        _take_one!(executor, out)
    end
    return out
end

capacity(executor::DistributedQueue) = executor.pending_exception === nothing ? executor.max_concurrency - executor.in_flight : 0
