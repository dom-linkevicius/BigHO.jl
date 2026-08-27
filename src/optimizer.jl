mutable struct Hyperoptimizer{S<:Sampler,F}
    params::Vector{Symbol}
    candidates::Tuple
    sampler::S
    objective::F
    n::Union{Int,Nothing}
    runs::Vector{RunEntry}
    completed::Vector{Int}
    n_pending::Int
    done::Bool
    best_min_id::Union{Int,Nothing}
    lock::ReentrantLock
end

"""
    Hyperoptimizer(objective, candidates::NamedTuple; sampler=RandomSampler(), n=nothing)

Construct a hyperparameter optimizer. `candidates` gives one [`Domain`](@ref)
per parameter name (e.g. `Continuous(1,5,0.1)`, `Nominal([true,false])`,
`Ordinal([1,2,5,10])`); `objective` is called as `objective(params...)`
unless wrapped in [`Stateful`](@ref). `n`, if given, bounds the total number
of trials. Only ever minimizes -- to maximize, minimize `-objective(...)`.
"""
function Hyperoptimizer(objective, candidates::NamedTuple; sampler::Sampler=RandomSampler(), n::Union{Int,Nothing}=nothing)
    cands = values(candidates)
    all(d -> d isa Domain, cands) ||
        throw(ArgumentError("every candidate must be a Domain (Continuous/Nominal/Ordinal), got types: $(typeof.(cands))"))
    params = collect(Symbol, keys(candidates))
    ho = Hyperoptimizer(params, cands, sampler, objective, n,
                         RunEntry[], Int[], 0, false,
                         nothing, ReentrantLock())
    init!(sampler, AskContext(cands, 0, n))
    return ho
end

reached_target(ho::Hyperoptimizer) = _reached_target(ho.n, ho.runs)
_reached_target(::Nothing, runs) = false
_reached_target(n::Int, runs) = length(runs) >= n

"""
    settarget!(ho, n)

Raise the planned total number of trials to `n` -- how you resume a run:
`settarget!(ho, n)` then `run!(ho)` again. Only raising is allowed; lowering
throws `ArgumentError`.
"""
function settarget!(ho::Hyperoptimizer, n::Int)
    _check_target_increase(ho.n, n)
    ho.n = n
    @info "Hyperoptimizer target set to $(ho.n) trials"
    return ho
end

_check_target_increase(::Nothing, n::Int) = nothing
function _check_target_increase(old::Int, n::Int)
    n < old && throw(ArgumentError("settarget!: new target ($n) is less than the current target ($old) -- settarget! can only raise the target"))
    return nothing
end

"""
    ask(ho) -> RunEntry

Draw the next candidate from `ho.sampler` and register a `Pending`
[`RunEntry`](@ref).
"""
function ask(ho::Hyperoptimizer)
    lock(ho.lock) do
        exhausted(ho.sampler, ho) && error("Hyperoptimizer's sampler is exhausted: no more candidates available")
        ctx = AskContext(ho.candidates, length(ho.runs), ho.n)
        raw = ho.sampler(ctx)
        id = length(ho.runs) + 1
        params = NamedTuple{Tuple(ho.params)}(Tuple(raw)) # e.g. (a = 1.5, b = true) -- labeled everywhere, not just in warnings
        entry = RunEntry(id, params)
        push!(ho.runs, entry)
        ho.n_pending += 1
        return entry
    end
end

function update_best!(ho::Hyperoptimizer, entry::RunEntry)
    if _is_new_best(ho.best_min_id, ho.runs, entry.value)
        ho.best_min_id = entry.id
    end
    return ho
end

# No NaN handling needed here: apply_outcome already excludes NaN outcomes
# as Failed, so a Completed entry's value is never NaN by the time it's ranked.
_is_new_best(::Nothing, runs, value) = true
_is_new_best(best_min_id::Int, runs, value) = value < runs[best_min_id].value

"""
    tell!(ho, entry, outcome)

Record `outcome` (the objective's return value, or a caught exception) for
`entry`, classifying it via [`apply_outcome`](@ref) and updating the cached
optimum. The sole mutation point for `ho.runs`/`ho.completed`.
"""
function tell!(ho::Hyperoptimizer, entry::RunEntry, outcome)
    lock(ho.lock) do
        told = apply_outcome(ho.runs[entry.id], outcome)
        ho.runs[told.id] = told
        ho.n_pending -= 1
        if told.status === Completed
            push!(ho.completed, told.id)
            update_best!(ho, told)
        end
        on_tell!(ho.sampler, told)
        return ho
    end
end

"""
    run!(ho; executor=Serial())

Drive `ho` to completion (until `ho.n` trials have completed, the sampler is
exhausted, or the run is interrupted), dispatching evaluations through
`executor`. To resume a finished run, call `settarget!(ho, n)` then `run!`
again.
"""
function run!(ho::Hyperoptimizer; executor::AbstractExecutor=Serial())
    start!(executor, ho)
    try
        while true
            while !ho.done && !reached_target(ho) && !exhausted(ho.sampler, ho) && capacity(executor) > 0
                entry = ask(ho)
                submit!(executor, entry, ho.objective)
            end
            for (entry, outcome) in poll(executor)
                tell!(ho, entry, outcome)
            end
            (ho.done || reached_target(ho) || exhausted(ho.sampler, ho)) && ho.n_pending == 0 && break
        end
    catch e
        _handle_run_error(e, ho)
    finally
        shutdown!(executor)
    end
    return ho
end

_handle_run_error(e, ho::Hyperoptimizer) = rethrow(e)
function _handle_run_error(::InterruptException, ho::Hyperoptimizer)
    ho.done = true
    @info "Aborting hyperoptimization, returning partial results"
end
