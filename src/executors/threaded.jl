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
end
function Threaded(max_concurrency::Int=Threads.nthreads())
    max_concurrency >= 1 || throw(ArgumentError("max_concurrency must be >= 1, got $max_concurrency"))
    Threads.nthreads() == 1 &&
        @warn "Threaded executor constructed with only 1 Julia thread available; trials will run concurrently as tasks but not in true parallel -- start Julia with `--threads=N` for real concurrency"
    return Threaded(max_concurrency, Channel{Tuple{RunEntry,Any}}(Inf), 0)
end

function start!(executor::Threaded, ho)
    executor.results = Channel{Tuple{RunEntry,Any}}(Inf)
    executor.in_flight = 0
    return nothing
end

shutdown!(::Threaded) = nothing

function submit!(executor::Threaded, entry::RunEntry, f)
    executor.in_flight += 1
    Threads.@spawn begin
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
    return nothing
end

# Blocks for the first result only if one is actually in flight (per
# AbstractExecutor's `poll` contract); drains any others already sitting in
# the channel so run! doesn't call back in for results that are ready now.
function poll(executor::Threaded)
    executor.in_flight == 0 && return Tuple{RunEntry,Any}[]
    # An InterruptException forwarded from submit! is re-thrown here, on the
    # driver's own call stack, where run!'s try/catch is watching for it --
    # rethrow() wouldn't work (there's no active catch block at this point),
    # but throw() on the same exception value raises it just as validly.
    function take_one!()
        entry, outcome = take!(executor.results)
        executor.in_flight -= 1
        outcome isa InterruptException && throw(outcome)
        return (entry, outcome)
    end
    # Explicitly typed -- a bare `[take_one!()]` infers its element type from
    # this specific tuple's concrete runtime type (e.g. with an ObjectiveOutcome
    # payload), and then errors on `push!` the moment a differently-typed
    # payload (e.g. a caught ErrorException) shows up later in the same batch.
    out = Tuple{RunEntry,Any}[take_one!()]
    while executor.in_flight > 0 && isready(executor.results)
        push!(out, take_one!())
    end
    return out
end

capacity(executor::Threaded) = executor.max_concurrency - executor.in_flight
