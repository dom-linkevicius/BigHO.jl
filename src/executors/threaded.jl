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
    Threads.@spawn put!(executor.results, (entry, safe_call(f, entry.params, entry.pre_artefact)))
    return nothing
end

# Blocks for the first result only if one is actually in flight (per
# AbstractExecutor's `poll` contract); drains any others already sitting in
# the channel so run! doesn't call back in for results that are ready now.
function poll(executor::Threaded)
    executor.in_flight == 0 && return Tuple{RunEntry,Any}[]
    # Explicitly typed -- a bare `[take!(...)]` infers its element type from
    # this specific tuple's concrete runtime type (e.g. with an ObjectiveOutcome
    # payload), and then errors on `push!` the moment a differently-typed
    # payload (e.g. a caught ErrorException) shows up later in the same batch.
    out = Tuple{RunEntry,Any}[take!(executor.results)]
    executor.in_flight -= 1
    while executor.in_flight > 0 && isready(executor.results)
        push!(out, take!(executor.results))
        executor.in_flight -= 1
    end
    return out
end

capacity(executor::Threaded) = executor.max_concurrency - executor.in_flight
